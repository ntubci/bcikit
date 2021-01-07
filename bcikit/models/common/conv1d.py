# -*- coding: utf-8 -*-
"""Common 1D convolutions
"""
import torch
from torch import nn
import torch.nn.functional as F

from typing import List
import math

    
class DepthWiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=in_channels, bias=bias)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)
