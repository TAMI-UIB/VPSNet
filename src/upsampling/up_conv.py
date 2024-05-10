import torch.nn as nn
from torch import Tensor

class UpsamplingConv(nn.Module):

    def __init__(self, n_channels, sampling_factor) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=sampling_factor, stride = sampling_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(x)
    
class DownsamplingConv(nn.Module):

    def __init__(self, n_channels, sampling_factor) -> None:
        super().__init__()
        self.down = nn.Conv2d(n_channels, n_channels, kernel_size=sampling_factor, stride = sampling_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)