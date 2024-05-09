import torch.nn as nn

class ProxNet(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks):
        super().__init__()
        self.blocks = n_blocks
        self.layers = nn.Sequential(
            *[ResBlock(out_channels, out_channels) for _ in range(n_blocks)])

        self.preconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) if in_channels != out_channels else None

    def forward(self, x):
        
        if self.preconv is not None:
            x = self.preconv(x)

        out = self.layers(x)

        if self.blocks > 1:
            out = out + x

        return out

class ResBlock(nn.Module):

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
