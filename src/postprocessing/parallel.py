import torch
import torch.nn as nn
from src.postprocessing.spectral_resnet import SpectralResNet
from src.postprocessing.spatial_resnet import ResNet

class Parallel(nn.Module):

    def __init__(self,n_channels, n_features=32, reduction=4, preprocessing = False) -> None:
        super().__init__()

        self.spectral = SpectralResNet(n_channels, n_features, reduction, preprocessing)
        self.spatial = ResNet(n_channels, n_channels)
        self.mix = nn.Conv2d(2*n_channels, n_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out_spectral = self.spectral(x)
        out_spatial = self.spatial(x)
        out = self.mix(torch.concat((out_spectral, out_spatial), dim=-3))
        return out
