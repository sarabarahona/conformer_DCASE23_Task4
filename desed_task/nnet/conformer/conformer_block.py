import torch
from .attention import RelMultiHeadAttn
from .convolution import ConvolutionModule
from .macaron_feed_forward import MacaronFeedForward

from .eff_modules import (
    FeedForwardModule,
    MultiHeadSelfAttentionModule,
    ConvolutionModuleDown
)

from .eff_layers import (
    Conv1d,
    Transpose
)


class ConformerBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout, kernel_size):
        super(ConformerBlock, self).__init__()
        self.ffn1 = MacaronFeedForward(d_model, d_ff, dropout)
        self.mhsa = RelMultiHeadAttn(n_head, d_model, dropout)
        self.conv = ConvolutionModule(d_model, dropout, kernel_size)
        self.ffn2 = MacaronFeedForward(d_model, d_ff, dropout)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = 0.5 * self.ffn1(x) + x
        x = x.permute(1, 0, 2)  # (B, T, D)
        x = self.mhsa(x, mask) + x  # (T, B, D)
        x = x.permute(1, 0, 2)  # (B, T, D)
        x = self.conv(x) + x
        x = 0.5 * self.ffn2(x) + x
        x = self.norm(x)
        return x, mask


class ConformerBlockEff(torch.nn.Module):

    def __init__(
        self, 
        dim_model, 
        dim_expand, 
        ff_ratio, 
        num_heads, 
        kernel_size, 
        att_group_size, 
        att_kernel_size,
        linear_att,
        Pdrop, 
        relative_pos_enc, 
        max_pos_encoding, 
        conv_stride,
        att_stride,
        causal
    ):
        super(ConformerBlockEff, self).__init__()

        # Feed Forward Module 1
        self.feed_forward_module1 = MacaronFeedForward(
            d_model=dim_model, 
            d_ff=dim_model * ff_ratio,
            dropout=Pdrop
        )

        # Multi-Head Self-Attention Module
        self.multi_head_self_attention_module = MultiHeadSelfAttentionModule(
            dim_model=dim_model, 
            num_heads=num_heads,  
            Pdrop=Pdrop, 
            max_pos_encoding=max_pos_encoding,
            relative_pos_enc=relative_pos_enc, 
            causal=causal,
            group_size=att_group_size,
            kernel_size=att_kernel_size,
            stride=att_stride,
            linear_att=linear_att
        )

        # Convolution Module
        self.convolution_module = ConvolutionModuleDown(
            dim_model=dim_model,
            dim_expand=dim_expand,
            kernel_size=kernel_size, 
            Pdrop=Pdrop, 
            stride=conv_stride,
            padding="causal" if causal else "same"
        )

        # Feed Forward Module 2
        self.feed_forward_module2 = MacaronFeedForward(
            d_model=dim_expand, 
            d_ff=dim_expand * ff_ratio,
            dropout=Pdrop
        )

        # Block Norm
        self.norm = torch.nn.LayerNorm(dim_expand, eps=1e-6)

        # Attention Residual
        self.att_res = torch.nn.Sequential(
            Transpose(1, 2),
            torch.nn.MaxPool1d(kernel_size=1, stride=att_stride),
            Transpose(1, 2)
        ) if att_stride > 1 else torch.nn.Identity()

        # Convolution Residual
        self.conv_res = torch.nn.Sequential(
            Transpose(1, 2),
            Conv1d(dim_model, dim_expand, kernel_size=1, stride=conv_stride),
            Transpose(1, 2)
        ) if dim_model != dim_expand else torch.nn.Sequential(
            Transpose(1, 2),
            torch.nn.MaxPool1d(kernel_size=1, stride=conv_stride),
            Transpose(1, 2)
        ) if conv_stride > 1 else torch.nn.Identity()

        # Stride
        self.stride = conv_stride * att_stride

    def forward(self, x, mask=None, hidden=None):

        # FFN Module 1
        x = x + 1/2 * self.feed_forward_module1(x)
        # MHSA Module
        x_att, attention, hidden = self.multi_head_self_attention_module(x, mask, hidden)
        x = self.att_res(x) + x_att
        # Conv Module
        x = self.conv_res(x) + self.convolution_module(x)
        # FFN Module 2
        x = x + 1/2 * self.feed_forward_module2(x)
        # Block Norm
        x = self.norm(x)

        return x, attention, hidden
