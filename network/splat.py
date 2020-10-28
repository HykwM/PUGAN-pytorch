"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU, Conv1d
from torch.nn.modules.utils import _pair

__all__ = ['SplAtconv2d']

class SplAtconv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=2, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtconv2d, self).__init__()
        #padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 2)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = nn.Linear(in_channels, channels * radix)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = nn.Linear(channels, inter_channels)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Linear(inter_channels, channels*radix)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)                #[64, 128, 8, 8] <- [64, 64, 8, 8] | [32, 16, 96] <- [32, 16, 16]
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch = x.shape[0]  #32
        rchannel = x.shape[2]   #96
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=2)   #[[64, 64, 8, 8][64, 64, 8, 8]] | [[32, 16, 48],[32, 16, 48]]
            gap = sum(splited)          #[32,16,48]
        else:
            gap = x
        #gap = F.adaptive_avg_pool1d(gap, 1)     #[32, 16, 1]
        gap = self.fc1(gap)     # <- [32, 16, 32]

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)       #[32, 16, 96]
        atten = self.rsoftmax(atten).view(batch, 1, -1)      #[32, 1536, 1]

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=2)    #[64, 64, 1, 1][64, 64, 1, 1] | [[32, 48, 1]*32]
            out = sum([att*split for (att, split) in zip(attens, splited)]) #[64, 64, 8, 8]
        else:
            out = atten * x

        out = out.permute((1, 0, 2))
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2) #[64, 2, 1, 64] <- [64, 128, 1, 1] |[32, 2, 1, 768] <- [32, 16, 96]
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)    #[64, 128]
        else:
            x = torch.sigmoid(x)
        return x

