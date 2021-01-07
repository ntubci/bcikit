# -*- coding: utf-8 -*-
"""Common 2D convolutions
"""
import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from typing import List
import math


class Conv2d(nn.Module):
    """
    Input: 4-dim tensor
        Shape [batch, in_channels, H, W]
    Return: 4-dim tensor
        Shape [batch, out_channels, H, W]
        
    Args:
        in_channels : int
            Should match input `channel`
        out_channels : int
            Return tensor with `out_channels`
        kernel_size : int or 2-dim tuple
        stride : int or 2-dim tuple, default: 1
        padding : int or 2-dim tuple or True
            Apply `padding` if given int or 2-dim tuple. Perform TensorFlow-like 'SAME' padding if True
        dilation : int or 2-dim tuple, default: 1
        groups : int or 2-dim tuple, default: 1
        w_in: int, optional
            The size of `W` axis. If given, `w_out` is available.
    
    Usage:
        x = torch.randn(1, 22, 1, 256)
        conv1 = Conv2dSamePadding(22, 64, kernel_size=17, padding=True, w_in=256)
        y = conv1(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="SAME", dilation=1, groups=1, w_in=None):
        super().__init__()
        
        padding = padding
        self.kernel_size = kernel_size = kernel_size
        self.stride = stride = stride
        self.dilation = dilation = dilation
        
        self.padding_same = False
        if padding == "SAME":
            self.padding_same = True
            padding = (0,0)
        
        if isinstance(padding, int):
            padding = (padding, padding)
            
        if isinstance(kernel_size, int):
            self.kernel_size = kernel_size = (kernel_size, kernel_size)
            
        if isinstance(stride, int):
            self.stride = stride = (stride, stride)
        
        if isinstance(dilation, int):
            self.dilation = dilation = (dilation, dilation)
            
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0 if padding==True else padding, 
            dilation=dilation, 
            groups=groups
        )
        
        if w_in is not None:
            self.w_out = int( ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1) / 1) + 1 )
        if self.padding_same == "SAME": # if SAME, then replace, w_out = w_in, obviously
            self.w_out = w_in
            
    def forward(self, x):
        if self.padding_same == True:
            x = self.pad_same(x, self.kernel_size, self.stride, self.dilation)
        return self.conv(x)
    
    # Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
    def get_same_padding(self, x: int, k: int, s: int, d: int):
        return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

    # Dynamically pad input x with 'SAME' padding for conv with specified args
    def pad_same(self, x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
        ih, iw = x.size()[-2:]
        pad_h, pad_w = self.get_same_padding(ih, k[0], s[0], d[0]), self.get_same_padding(iw, k[1], s[1], d[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
        return x


class Conv2dBlockELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, activation=nn.ELU, w_in=None):
        super(Conv2dBlockELU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

        if w_in is not None:
            self.w_out = int( ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1) / 1) + 1 )

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth=1, padding=0, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
            
        if isinstance(kernel_size, tuple):
            padding = (
                kernel_size[0]//2 if kernel_size[0]-1 != 0 else 0,
                kernel_size[1]//2 if kernel_size[1]-1 != 0 else 0
            )
            
        self.depthwise = DepthwiseConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
