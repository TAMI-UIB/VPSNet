import torch
import torch.nn as nn

class EdgeProtector(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size, features=3):
        super(EdgeProtector, self).__init__()
        init_channels = x_channels + y_channels
        mid_channels = x_channels + y_channels + features
        final_channels = x_channels
        self.convbatchnorm1 = CBNReLU(in_channels=init_channels, out_channels=mid_channels,
                                                kernel_size=kernel_size)
        self.convbatchnorm2 = CBNReLU(in_channels=mid_channels, out_channels=mid_channels,
                                                kernel_size=kernel_size)
        self.convbatchnorm3 = CBNReLU(in_channels=mid_channels, out_channels=final_channels,
                                                kernel_size=kernel_size)

    def forward(self, x, y):
        features_1 = self.convbatchnorm1(torch.cat((x, y), dim=1))
        features_2 = self.convbatchnorm2(features_1)
        features_3 = self.convbatchnorm3(features_2)
        features_3 = features_3 + x
        return features_3

class CBNReLU(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_channels, out_channels, kernel_size):
        super(CBNReLU, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels, eps=1e-05),
            nn.ReLU(inplace=False),
        )
        self.apply(self.init_xavier)

    def forward(self, x):
        x = self.layers(x)
        return x

    def init_xavier(self, module):
        if type(module) == nn.Conv2d:
            nn.init.xavier_uniform_(module.weight)