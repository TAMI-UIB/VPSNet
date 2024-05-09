from .parallel import Parallel
from .series import Series
from .spatial_resnet import ResNet
from .spectral_resnet import SpectralResNet, SpectralResNetBlocks

dict_post = {
    "spectral": SpectralResNet,
    "spectralBlocks": SpectralResNetBlocks,
    "spatial": ResNet,
    "parallel": Parallel,
    "series": Series,
}