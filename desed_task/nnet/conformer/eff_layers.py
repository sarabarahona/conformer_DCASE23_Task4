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

# Activation Functions
from .activation import (
    Swish
)

###############################################################################
# Layers
###############################################################################

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


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)