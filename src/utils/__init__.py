from .losses import *
from ..upsampling.upsamplings import *
from torch.optim import Adam


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