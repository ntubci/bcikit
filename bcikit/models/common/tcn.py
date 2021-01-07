# -*- coding: utf-8 -*-
"""TemporalConvNet
"""
from torch import nn


class TemporalConvNet(nn.Module):
    """
    TCN layer
    An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
    https://arxiv.org/abs/1803.01271
    https://github.com/locuslab/TCN
    
    Usage: 
        tcn = TemporalConvNet(
            num_channels=11
        )
        x = torch.randn(2, 11, 250)
        print("Input shape:", x.shape)
        y = tcn(x)
        print("Output shape:", y.shape)

    """
    def __init__(self, num_channels, kernel_size=7, dropout=0.1, nhid=32, levels=8):
        super(TemporalConvNet, self).__init__()
        
        channel_sizes = [nhid] * levels
        
        layers = []
        num_levels = len(channel_sizes)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels if i == 0 else channel_sizes[i-1]
            out_channels = channel_sizes[i]
            layers += [
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                              padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    
    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
