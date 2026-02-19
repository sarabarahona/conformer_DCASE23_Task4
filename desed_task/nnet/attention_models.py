import torch
import warnings
import pdb


from .fdyCNN import FDY_CNN, CNN
from .light_fdyCNN import light_FDY_CNN 
from .conformer.conformer_encoder import ConformerEncoder, ConformerEncoderEff
from .transformer.encoder import Encoder as TransformerEncoder


class AttModel(torch.nn.Module):
    def __init__(
        self,
        n_class,
        cnn_type="cnn",
        cnn_kwargs=None,
        dim_freq=1,
        encoder_type="Conformer",
        encoder_kwargs=None,
        pooling="token",
        layer_init="pytorch",
        use_embeddings=False,
        embedding_size=527,
        embedding_type="global",
        frame_emb_enc_dim=512,
        aggregation_type="global",
    ):
        super(AttModel, self).__init__()

        self.pooling = pooling
        self.dim_freq = dim_freq
        self.use_embeddings = use_embeddings
        self.embedding_type = embedding_type
        self.aggregation_type = aggregation_type

        if cnn_type == 'cnn':
            self.cnn = CNN(n_in_channel=1, **cnn_kwargs)
        elif cnn_type == 'freq':
            self.cnn = FDY_CNN(n_in_channel=1, **cnn_kwargs)
        elif cnn_type =="light_freq":
            self.cnn = light_FDY_CNN(**cnn_kwargs)

        input_dim = self.cnn.nb_filters[-1]*self.dim_freq
        if isinstance(encoder_kwargs["dim_model"], int):
            adim = encoder_kwargs["dim_model"]
        else: 
            adim = encoder_kwargs["dim_model"][-1]

        if encoder_type == "Transformer":
            self.encoder = TransformerEncoder(input_dim, **encoder_kwargs)
        elif encoder_type == "Conformer":
            self.encoder = ConformerEncoder(input_dim, **encoder_kwargs)
        elif encoder_type == "ConformerEff":
            self.encoder = ConformerEncoderEff(input_dim, encoder_kwargs)
        else:
            raise ValueError("Choose encoder_type in ['Transformer', 'Conformer', 'ConformerEff']")

        self.classifier = torch.nn.Linear(adim, n_class)

        if self.pooling == "attention":
            self.dense = torch.nn.Linear(adim, n_class)
            self.sigmoid = torch.nn.Sigmoid()
            self.softmax = torch.nn.Softmax(dim=-1)

        elif self.pooling == "token":
            self.linear_emb = torch.nn.Linear(1, input_dim)
            self.sigmoid = torch.nn.Sigmoid()

        if self.use_embeddings:
            if self.aggregation_type == "frame":
                self.frame_embs_encoder = torch.nn.GRU(batch_first=True, input_size=embedding_size,
                                                      hidden_size=512,
                                                      bidirectional=True)
                self.shrink_emb = torch.nn.Sequential(torch.nn.Linear(2 * frame_emb_enc_dim, input_dim),
                                                      torch.nn.LayerNorm(input_dim))
                self.cat_tf = torch.nn.Linear(2*input_dim, input_dim)
            elif self.aggregation_type == "global":
                self.shrink_emb = torch.nn.Sequential(torch.nn.Linear(embedding_size, input_dim),
                                                      torch.nn.LayerNorm(input_dim))
                self.cat_tf = torch.nn.Linear(2*input_dim, input_dim)
            elif self.aggregation_type == "interpolate":
                self.cat_tf = torch.nn.Linear(input_dim+embedding_size, input_dim)
            elif self.aggregation_type == "pool1d":
                self.cat_tf = torch.nn.Linear(input_dim+embedding_size, input_dim)
            else:
                self.cat_tf = torch.nn.Linear(2*input_dim, input_dim)

        self.reset_parameters(layer_init)

    def forward(self, x, mask=None, embeddings=None):

        #Permute number of frames with mels
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.cnn(x) #[bs, channels, frames, mels]

        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)  # [bs, frames, chan]
        
        
        if self.use_embeddings:
            if self.aggregation_type == "global":
                x = self.cat_tf(torch.cat((x, self.shrink_emb(embeddings).unsqueeze(1).repeat(1, x.shape[1], 1)), -1))
            elif self.aggregation_type == "frame":
                # there can be some mismatch between seq length of cnn of crnn and the pretrained embeddings, we use an rnn
                # as an encoder and we use the last state
                last, _ = self.frame_embs_encoder(embeddings.transpose(1, 2))
                embeddings = last[:, -1]
                x = self.cat_tf(torch.cat((x, self.shrink_emb(embeddings).unsqueeze(1).repeat(1, x.shape[1], 1)), -1))
            elif self.aggregation_type == "interpolate":
                output_shape = (embeddings.shape[1], x.shape[1])
                reshape_emb = torch.nn.functional.interpolate(embeddings.unsqueeze(1), size=output_shape, mode='nearest-exact').squeeze(1).transpose(1, 2)
                x = self.cat_tf(torch.cat((x, reshape_emb), -1))
            elif self.aggregation_type == "pool1d":
                reshape_emb = torch.nn.functional.adaptive_avg_pool1d(embeddings, x.shape[1]).transpose(1, 2)
                x = self.cat_tf(torch.cat((x, reshape_emb), -1))
            else:
                pass
        if self.pooling == "token":
            tag_token = self.linear_emb(torch.ones(x.size(0), 1, 1).to(x))
            x = torch.cat([tag_token, x], dim=1)

        x, _ = self.encoder(x, mask = mask)[0:2]

        if self.pooling == "attention":
            #Strong predictions
            strong = self.classifier(x)
            strong = self.sigmoid(strong)
            #Weak predictions
            sof = self.dense(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
            # Convert to logit to calculate loss with bcelosswithlogits
            weak = torch.log(weak / (1 - weak))
        elif self.pooling == "token":
            x = self.classifier(x)
            #Add sigmoid here
            x = self.sigmoid(x)
            weak = x[:, 0, :]
            strong = x[:, 1:, :]
        elif self.pooling == "auto":
            strong = self.classifier(x)
            weak = self.autopool(strong)

        strong = strong.permute(0, 2, 1)

        return strong, weak

    def reset_parameters(self, initialization: str = "pytorch"):
        if initialization.lower() == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if initialization.lower() == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif initialization.lower() == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif initialization.lower() == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif initialization.lower() == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError(f"Unknown initialization: {initialization}")
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()
