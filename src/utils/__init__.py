from .losses import *
from .upsamplings import *
from torch.optim import Adam

from .histogram import HistogramMatching, HistogramEqualization

default_hp = {
    "learning_rate": 1e-3,
    "kernel_size": 9,
    "std": 1.7
}

dict_losses = {
    "MSE": MSELoss(reduction='mean'),
    "L1": L1Loss(reduction="mean"),
    "MSEL1": MSEL1Loss(),
    "Radiometric": RadiometricLoss(),
    "RadiometricMSE": RadiometricMSELoss(),
    "RadiometricL1": RadiometricL1Loss(),
    "RadiometricPerStage": RadiometricPerStage(),
    "NLRNetLoss": NLRNetLoss(),
    "L1SAMLoss": L1SAMLoss(),
    "L1PerStage": L1PerStage()
}

dict_optimizers = {
    "Adam": Adam
}

dict_upsamplings = {
    "bicubic": [Upsampling, Downsampling],
    "pgcu": [Upsampling, Downsampling],
    "zero-padding": [Upsampling0, Downsampling0],
    "conv": [UpsamplingConv, DownsamplingConv],
    "malisat": [UpSamp_4_2, Downsamp_4_2],
    "malisat2": [UpSamp_2_2, Downsamp_2_2],
    "pixel_shuffle": [PSUpsampling, PSDownsampling]
}

dict_histograms = {
    "LHM": HistogramMatching(),
    "HM": HistogramMatching(local_flag=False),
    "CLAHE": HistogramEqualization()
}
