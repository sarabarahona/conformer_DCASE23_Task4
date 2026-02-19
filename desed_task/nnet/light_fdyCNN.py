import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pandas as pd
from copy import deepcopy
#from conformer.encoder import ConformerBlock as ConformerBlock2



class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, in_dim):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(lin)
        res = x * sig
        # ores = x * sig
        return res


class BiGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        #self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return x

########################################################################################################################
#                                                Squeeze and Excitation                                                #
########################################################################################################################
class SELayer(nn.Module):
    def __init__(self, dim, reduction=16, attend_dim="chan"):
        super(SELayer, self).__init__()
        self.attend_dim = attend_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hid_dim = dim // reduction

        if hid_dim < 4:
            hid_dim = 4

        if attend_dim == "chan-freq":
            self.fc = nn.Sequential(nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(8, 1, 3, stride=1, padding=1, bias=False),
                                    nn.Sigmoid())

        else:
            self.fc = nn.Sequential(nn.Linear(dim, hid_dim, bias=False),
                                    # nn.BatchNorm1d(hid_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hid_dim, dim, bias=False),
                                    nn.Sigmoid())

    def forward(self, x):   #x size : [bs, chan, frames, freqs]
        b, c, t, f = x.size()
        if self.attend_dim == "chan":
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)

        elif self.attend_dim == "chan_timewise":
            y = torch.mean(x, dim=3).transpose(1, 2)  #x size : [bs, frames, chan]
            y = self.fc(y).transpose(1, 2).view(b, c, t, 1)

        elif self.attend_dim == "freq":
            y = torch.mean(x, dim=(1, 2))
            y = self.fc(y).view(b, 1, 1, f)

        elif self.attend_dim == "freq_timewise":
            y = torch.mean(x, dim=1)                  #x size : [bs, frames, freqs]
            y = self.fc(y).view(b, 1, t, f)

        elif self.attend_dim == "chan-freq":
            y = torch.mean(x, dim=2).view(b, 1, c, f)
            y = self.fc(y).view(b, c, 1, f)

        return x * y.expand_as(x)


class light_FDY_CNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 activation="Relu",
                 dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 SE_layers=[0, 0, 0, 0, 0, 0, 0],
                 se_reduction=16,
                 attend_dim='chan',
                 SE2_layers=[0, 0, 0, 0, 0, 0, 0],
                 se2_reduction=16,
                 attend_dim2='chan'):
        super(light_FDY_CNN, self).__init__()
        self.nb_filters = nb_filters
        self.nb_filters_last = nb_filters[-1]
        cnn = nn.Sequential()
        freq_dims = [128, 64, 32, 16, 8, 4, 2]
        time_dims = [625, 323, 323, 323, 323, 323, 323]

        def conv(i, normalization="batch", dropout=None, activ='relu'):
            in_dim = n_input_ch if i == 0 else nb_filters[i - 1]
            out_dim = nb_filters[i]
            # convolution
            cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))
            # normalization
            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))
            # non-linearity
            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))
            # squeeze-excitation
            if SE_layers[i] == 1:
                if attend_dim in ["freq", "freq_timewise"]:
                    se_dim = freq_dims[i]
                else:
                    se_dim = out_dim
                cnn.add_module("SElayer{0}".format(i), SELayer(se_dim, reduction=se_reduction, attend_dim=attend_dim))
            if SE2_layers[i] == 1:
                if attend_dim2 in ["freq", "freq_timewise"]:
                    se_dim2 = freq_dims[i]
                else:
                    se_dim2 = out_dim
                cnn.add_module("SElayer2{0}".format(i), SELayer(se_dim2, reduction=se2_reduction,
                                                                attend_dim=attend_dim2))
            # dropout
            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(nb_filters)):
            conv(i, normalization=normalization, dropout=dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x):    #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        return x