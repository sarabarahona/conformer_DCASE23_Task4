# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Attentions
from .eff_attention import (
    # Abs Attentions
    MultiHeadAttention,
    GroupedMultiHeadAttention,
    LocalMultiHeadAttention,
    StridedMultiHeadAttention,
    StridedLocalMultiHeadAttention,
    MultiHeadLinearAttention,
    # Rel Attentions
    RelPosMultiHeadSelfAttention,
    GroupedRelPosMultiHeadSelfAttention,
    LocalRelPosMultiHeadSelfAttention,
    StridedRelPosMultiHeadSelfAttention,
    StridedLocalRelPosMultiHeadSelfAttention
)

# Activations
from .activation import (
    Swish,
    Glu
)

###############################################################################
# Conformer Modules
###############################################################################

class FeedForwardModule(nn.Module):

    """Transformer Feed Forward Module

    Args:
        dim_model: model feature dimension
        dim_ffn: expanded feature dimension
        Pdrop: dropout probability
        act: inner activation function
        inner_dropout: whether to apply dropout after the inner activation function

    Input: (batch size, length, dim_model)
    Output: (batch size, length, dim_model)
    
    """

    def __init__(self, dim_model, dim_ffn, Pdrop, act, inner_dropout):
        super(FeedForwardModule, self).__init__()

        # Assert
        assert act in ["relu", "swish"]

        # Layers
        self.layers = nn.Sequential(
            nn.LayerNorm(dim_model, eps=1e-6),
            nn.Linear(dim_model, dim_ffn),
            Swish() if act=="swish" else nn.ReLU(),
            nn.Dropout(p=Pdrop) if inner_dropout else nn.Identity(),
            nn.Linear(dim_ffn, dim_model),
            nn.Dropout(p=Pdrop)
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadSelfAttentionModule(nn.Module):

    """Multi-Head Self-Attention Module

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        Pdrop: residual dropout probability
        max_pos_encoding: maximum position
        relative_pos_enc: whether to use relative postion embedding
        causal: True for causal attention with masked future context
        group_size: Attention group size
        kernel_size: Attention kernel size
        stride: Query stride
        linear_att: whether to use multi-head linear self-attention

    """

    def __init__(self, dim_model, num_heads, Pdrop, max_pos_encoding, relative_pos_enc, causal, group_size, kernel_size, stride, linear_att):
        super(MultiHeadSelfAttentionModule, self).__init__()

        # Assert
        assert not (group_size > 1 and kernel_size is not None), "Local grouped attention not implemented"
        assert not (group_size > 1 and stride > 1 is not None), "Strided grouped attention not implemented"
        assert not (linear_att and relative_pos_enc), "Linear attention requires absolute positional encodings"

        # Pre Norm
        self.norm = nn.LayerNorm(dim_model, eps=1e-6)

        # Multi-Head Linear Attention
        if linear_att:
            self.mhsa = MultiHeadLinearAttention(dim_model, num_heads)

        # Grouped Multi-Head Self-Attention
        elif group_size > 1:
            if relative_pos_enc:
                self.mhsa = GroupedRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding, group_size)
            else:
                self.mhsa = GroupedMultiHeadAttention(dim_model, num_heads, group_size)
        
        # Local Multi-Head Self-Attention
        elif kernel_size is not None and stride == 1:
            if relative_pos_enc:
                self.mhsa = LocalRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, kernel_size)
            else:
                self.mhsa = LocalMultiHeadAttention(dim_model, num_heads, kernel_size)

        # Strided Multi-Head Self-Attention
        elif kernel_size is None and stride > 1:
            if relative_pos_enc:
                self.mhsa = StridedRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding, stride)
            else:
                self.mhsa = StridedMultiHeadAttention(dim_model, num_heads, stride)

        # Strided Local Multi-Head Self-Attention
        elif stride > 1 and kernel_size is not None:
            if relative_pos_enc:
                self.mhsa = StridedLocalRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, kernel_size, stride)
            else:
                self.mhsa = StridedLocalMultiHeadAttention(dim_model, num_heads, kernel_size, stride)

        # Multi-Head Self-Attention
        else:
            if relative_pos_enc:
                self.mhsa = RelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding)
            else:
                self.mhsa = MultiHeadAttention(dim_model, num_heads)
            
        # Dropout
        self.dropout = nn.Dropout(Pdrop)

        # Module Params
        self.rel_pos_enc = relative_pos_enc
        self.linear_att = linear_att

    def forward(self, x, mask=None, hidden=None):

        # Pre Norm
        x = self.norm(x)

        # Multi-Head Self-Attention
        if self.linear_att:
            x, attention = self.mhsa(x, x, x)
        elif self.rel_pos_enc:
            x, attention, hidden = self.mhsa(x, x, x, mask, hidden)
        else:
            x, attention = self.mhsa(x, x, x, mask)

        # Dropout
        x = self.dropout(x)

        return x, attention, hidden

################ Layers for ConvolutionModule ###############

class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class Conv1d(nn.Conv1d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride = 1, 
        padding = "same", 
        dilation = 1, 
        groups = 1, 
        bias = True
    ):
        super(Conv1d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0, 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding_mode="zeros")

        # Assert
        assert padding in ["valid", "same", "causal"]

        # Padding
        if padding == "valid":
            self.pre_padding = None
        elif padding == "same":
            self.pre_padding = nn.ConstantPad1d(padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2), value=0)
        elif padding == "causal":
            self.pre_padding = nn.ConstantPad1d(padding=(kernel_size - 1, 0), value=0)

        # Variational Noise
        self.noise = None
        self.vn_std = None

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(), device=self.weight.device, dtype=self.weight.dtype)

        # Broadcast Noise
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, input):

        # Weight
        weight = self.weight

        # Add Noise
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise

        # Padding
        if self.pre_padding is not None:
            input = self.pre_padding(input)

        # Apply Weight
        return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ConvolutionModuleDown(nn.Module):

    """Conformer Convolution Module

    Args:
        dim_model: input feature dimension
        dim_expand: output feature dimension
        kernel_size: 1D depthwise convolution kernel size
        Pdrop: residual dropout probability
        stride: 1D depthwise convolution stride
        padding: "valid", "same" or "causal"

    Input: (batch size, input length, dim_model)
    Output: (batch size, output length, dim_expand)
    
    """

    def __init__(self, dim_model, dim_expand, kernel_size, Pdrop, stride, padding):
        super(ConvolutionModuleDown, self).__init__()

        # Layers
        self.layers = nn.Sequential(
            nn.LayerNorm(dim_model, eps=1e-6),
            Transpose(1, 2),
            Conv1d(dim_model, 2 * dim_expand, kernel_size=1),
            Glu(dim=1),
            Conv1d(dim_expand, dim_expand, kernel_size, stride=stride, padding=padding, groups=dim_expand),
            nn.BatchNorm1d(dim_expand),
            Swish(),
            Conv1d(dim_expand, dim_expand, kernel_size=1),
            Transpose(1, 2),
            nn.Dropout(p=Pdrop)
        )

    def forward(self, x):
        return self.layers(x)
