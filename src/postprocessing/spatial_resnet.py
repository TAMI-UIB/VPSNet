import torch.nn as nn

class ResNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block1 = CBNReLU(in_channels, 32, 3)
        self.block2 = CBNReLU(32, 32, 3)
        self.block3 = CBNReLU(32, out_channels, 3)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out = out + x
        return out
    
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