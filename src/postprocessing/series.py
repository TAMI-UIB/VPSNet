import torch.nn as nn
from src.postprocessing.spectral_resnet import SpectralResNet
from src.postprocessing.spatial_resnet import ResNet

class Series(nn.Module):

    def __init__(self,n_channels, n_features=32, reduction=4, preprocessing = False) -> None:
        super().__init__()

        self.spectral = SpectralResNet(n_channels, n_features, reduction, preprocessing)
        self.spatial = ResNet(n_channels, n_channels)

    def forward(self, x):
        out_spectral = self.spectral(x)
        out = self.spatial(out_spectral)
        return out
