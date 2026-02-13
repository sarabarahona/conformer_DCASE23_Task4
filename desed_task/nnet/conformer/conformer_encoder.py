import torch
from .conformer_block import ConformerBlock, ConformerBlockEff
from .repeat import repeat
from ..transformer.embedding import PositionalEncoding
from .eff_modules import (
    FeedForwardModule,
    MultiHeadSelfAttentionModule,
    ConvolutionModuleDown
)
from .eff_attention import (
    SinusoidalPositionalEncoding,
    StreamingMask,
)

class ConformerEncoder(torch.nn.Module):

    def __init__(
        self,
        idim: int,
        adim: int = 144,
        dropout_rate: float = 0.1,
        elayers: int = 3,
        eunits: int = 576,
        aheads: int = 4,
        kernel_size: int = 7,
    ):
        super(ConformerEncoder, self).__init__()
        assert adim % aheads == 0
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(idim, adim),
            torch.nn.LayerNorm(adim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            PositionalEncoding(adim, dropout_rate),
        )        
        self.conformer_blocks = repeat(elayers, lambda: ConformerBlock(adim, eunits, aheads, dropout_rate, kernel_size))

    def forward(self, x, mask=None):
        x = self.input_layer(x)
        x, mask = self.conformer_blocks(x, mask)
        return x, mask

class ConformerEncoderEff(torch.nn.Module):

    def __init__(self, input_dim, params):
        super(ConformerEncoderEff, self).__init__()

        self.linear = torch.nn.Linear(input_dim, params["dim_model"][0] if isinstance(params["dim_model"], list) else  params["dim_model"])
        self.dropout = torch.nn.Dropout(p=params["dropout_rate"])
        self.pos_enc = None if params["relative_pos_enc"] else SinusoidalPositionalEncoding(params["max_pos_encoding"], params["dim_model"][0] if isinstance(params["dim_model"], list) else  params["dim_model"])

        self.blocks = torch.nn.ModuleList([ConformerBlockEff(
            dim_model=params["dim_model"][(block_id > torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["dim_model"], list) else params["dim_model"],
            dim_expand=params["dim_model"][(block_id >= torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["dim_model"], list) else params["dim_model"],
            ff_ratio=params["ff_ratio"],
            num_heads=params["num_heads"][(block_id > torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["num_heads"], list) else params["num_heads"], 
            kernel_size=params["kernel_size"][(block_id >= torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["kernel_size"], list) else params["kernel_size"], 
            att_group_size=params["att_group_size"][(block_id > torch.tensor(params.get("strided_blocks", []))).sum()] if isinstance(params.get("att_group_size", 1), list) else params.get("att_group_size", 1),
            att_kernel_size=params["att_kernel_size"][(block_id > torch.tensor(params.get("strided_layers", []))).sum()] if isinstance(params.get("att_kernel_size", None), list) else params.get("att_kernel_size", None),
            linear_att=params.get("linear_att", False),
            Pdrop=params["dropout_rate"], 
            relative_pos_enc=params["relative_pos_enc"], 
            max_pos_encoding=params["max_pos_encoding"] // params.get("stride", 2)**int((block_id > torch.tensor(params.get("strided_blocks", []))).sum()),
            conv_stride=(params["conv_stride"][(block_id > torch.tensor(params.get("strided_blocks", []))).sum()] if isinstance(params["conv_stride"], list) else params["conv_stride"]) if block_id in params.get("strided_blocks", []) else 1,
            att_stride=(params["att_stride"][(block_id > torch.tensor(params.get("strided_blocks", []))).sum()] if isinstance(params["att_stride"], list) else params["att_stride"]) if block_id in params.get("strided_blocks", []) else 1,
            causal=params.get("causal", False)
        ) for block_id in range(params["elayers"])])

    def forward(self, x, x_len = None, mask=None):

        # Input dimension (bs, frames, channel)
        x = self.linear(x)

        # Dropout
        x = self.dropout(x)

        # Sinusoidal Positional Encodings
        if self.pos_enc is not None:
            x = x + self.pos_enc(x.size(0), x.size(1))

        # Conformer Blocks
        attentions = []
        for block in self.blocks:
            x, attention, hidden = block(x, mask)
            attentions.append(attention) # attention scores

            # Strided Block
            if block.stride > 1:
                # Stride Mask (B, 1, T // S, T // S)
                if mask is not None:
                    mask = mask[:, :, ::block.stride, ::block.stride]
                if x_len is not None:
                    x_len = torch.div(x_len - 1, block.stride, rounding_mode='floor') + 1

        return x, x_len, attentions, mask
