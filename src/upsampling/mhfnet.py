import torch
import torch.nn as nn

from src.upsampling.edge_protector import EdgeProtector

class UpSamp_4_2(nn.Module):
    def __init__(self, n_channels, sampling_factor, kernel_size=3, depthwise_coef=1, features=3):
        super(UpSamp_4_2, self).__init__()
        self.conv2dTrans = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
        )
        self.conv2dTrans_factor2 = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=5,
            stride=2,
            padding=5 // 2,
            output_padding=1,
        )
        self.edge_protector1 = EdgeProtector(n_channels, 1, kernel_size, features=features)
        self.edge_protector2 = EdgeProtector(n_channels, 1, kernel_size, features=features)
        self.conv2d = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * depthwise_coef,
            kernel_size=3,
            padding=1,
            groups=n_channels,
        )
        self.conv2d.weight.data = (1 / 16) * torch.ones(self.conv2d.weight.data.size())

    def forward(self, input, support_d2, support_full):

        input = self.conv2dTrans(input)

        input = self.edge_protector1(input, support_d2 / 10)

        input = self.conv2dTrans_factor2(input)
        input = self.edge_protector2(input, support_full / 10)

        input = self.conv2d(input)

        return input

class Downsamp_4_2(nn.Module):
    def __init__(self, channels, sampling_factor,  depthwise_multiplier=1):
        super(Downsamp_4_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels * depthwise_multiplier, kernel_size=5,
                               padding=5 // 2, groups=channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels * depthwise_multiplier, kernel_size=5,
                               padding=5 // 2, groups=channels)

    def forward(self, input):
        height = input.size(2)
        width = input.size(3)
        # Downsampling by a factor of 2
        x_d2 = self.conv2(input)[:, :, 0:height: 2, 0:width: 2]
        # Downsampling by a ratio of 2
        x_low = self.conv1(x_d2)[:, :, 0: int(height / 2): 2, 1: int(width / 2): 2]

        return x_low
    
class UpSamp_2_2(nn.Module):
    def __init__(self, n_channels, sampling_factor, kernel_size=3, depthwise_coef=1, features=3):
        super(UpSamp_2_2, self).__init__()
        self.conv2dTrans = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
        )
        self.edge_protector = EdgeProtector(n_channels, 1, kernel_size, features=features)
        self.conv2d = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * depthwise_coef,
            kernel_size=3,
            padding=1,
            groups=n_channels,
        )
        self.conv2d.weight.data = (1 / 16) * torch.ones(self.conv2d.weight.data.size())

    def forward(self, input, support_d2, support_full):

        input = self.conv2dTrans(input)

        input = self.edge_protector(input, support_full / 10)

        input = self.conv2d(input)

        return input

class Downsamp_2_2(nn.Module):
    def __init__(self, channels, sampling_factor,  depthwise_multiplier=1):
        super(Downsamp_2_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels * depthwise_multiplier, kernel_size=5,
                               padding=5 // 2, groups=channels)
        

    def forward(self, input):
        height = input.size(2)
        width = input.size(3)
        
        # Downsampling by a factor of 2
        x_low = self.conv1(input)[:, :, 0:height: 2, 0:width: 2]

        return x_low