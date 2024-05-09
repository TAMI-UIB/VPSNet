import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, CosineSimilarity

from src.utils.metrics import SAM

from typing import List

class RadiometricLoss(nn.Module):

    def __init__(self) -> None:
        super(RadiometricLoss, self).__init__()

    def forward(self, u: Tensor, pan: Tensor, P_tilde: Tensor, u_tilde: Tensor) -> Tensor:

        radiometric = u_tilde*pan-u*P_tilde
        norm_per_channel = torch.linalg.matrix_norm(radiometric)
        loss = torch.linalg.vector_norm(norm_per_channel)**2

        return loss
    
class L1PerStage(nn.Module):

    def __init__(self, alpha = 1) -> None:
        super().__init__()
        self.alpha_stage = alpha
        self.l1 = L1Loss()

    def forward(self, pred, list_uk, target):
        loss_output = self.l1(pred, target)

        loss_stages = 0
        for output_stage in list_uk:
            loss_stages += self.l1(output_stage, target)
        
        return loss_output + self.alpha_stage*loss_stages


class RadiometricMSELoss(nn.Module):

    def __init__(self, alpha: float = 1e-2) -> None:
        super(RadiometricMSELoss, self).__init__()
        self.mse = MSELoss(reduction='mean')
        self.radiometric = RadiometricLoss()
        self.alpha = alpha

    def forward(self, u: Tensor, pan: Tensor, P_tilde: Tensor, u_tilde: Tensor, gt: Tensor) -> Tensor:

        radiometric_loss = self.radiometric(u, pan, P_tilde, u_tilde)
        mse_loss = self.mse(u, gt)

        loss = mse_loss + self.alpha * radiometric_loss

        return loss
    
class RadiometricL1Loss(nn.Module):

    def __init__(self, alpha: float = 1e-2) -> None:
        super(RadiometricL1Loss, self).__init__()
        self.l1 = L1Loss(reduction='mean')
        self.radiometric = RadiometricLoss()
        self.alpha = alpha

    def forward(self, u: Tensor, pan: Tensor, P_tilde: Tensor, u_tilde: Tensor, gt: Tensor) -> Tensor:

        radiometric_loss = self.radiometric(u, pan, P_tilde, u_tilde)
        l1_loss = self.l1(u, gt)

        loss = l1_loss + self.alpha * radiometric_loss

        return loss
    
class RadiometricPerStage(nn.Module):

    def __init__(self, alpha: float = 0.1, beta: float = 1e-3, loss = 'l1') -> None:
        super(RadiometricPerStage, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.isL1Stages = False

        match loss:
            case 'L1':
                self.loss = L1Loss(reduction='mean')
            case 'MSE':
                self.loss = MSELoss(reduction='mean')
            case 'L1PerStage':
                self.loss = L1PerStage(self.alpha)
                self.isL1Stages = True
            case _:
                self.loss = L1Loss(reduction='mean')

        self.radiometric = RadiometricLoss()
        

    def forward(self, pred: Tensor, list_uk, list_radiometrics, target: Tensor) -> Tensor:

        loss = self.loss(pred, list_uk, target) if self.isL1Stages else self.loss(pred, target)

        loss_final = loss + self.beta * torch.sum(torch.Tensor(list_radiometrics))

        return loss_final
    
class MSEL1Loss(nn.Module):

    def __init__(self, alpha: float = 1, beta: float = 1) -> None:
        super(MSEL1Loss, self).__init__()
        self.mse = MSELoss(reduction='mean')
        self.l1 = L1Loss(reduction='mean')
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:

        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        loss = self.alpha*mse_loss + self.beta * l1_loss

        return loss
    
class SpectralLoss(nn.Module):

    def __init__(self, alpha=1):
        self.sam = SAM
        self.base_loss = L1Loss(reduction='mean')
        self.alpha = alpha

    def forward(self, pred: Tensor, target: Tensor, spectral_reference: Tensor):
        return self.alpha*self.sam(pred, spectral_reference) + self.base_loss(pred, target)

class SpatialLoss(nn.Module):

    def __init__(self, alpha=1):
        self.mse = MSELoss(reduction='mean')
        self.base_loss = L1Loss(reduction='mean')
        self.alpha = alpha

    def forward(self, pred: Tensor, target: Tensor, spatial_reference: Tensor):
        return self.alpha*self.mse(pred, spatial_reference) + self.base_loss(pred, target)
    
class NLRNetLoss(nn.Module):

    def __init__(self) -> None:
        super(NLRNetLoss, self).__init__()
        self.mse = MSELoss(reduction='mean')
        self.l1 = L1Loss(reduction='mean')
        self.cosine = CosineSimilarity()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:

        spatital_loss = self.l1(pred, target) * 85
        spectral_loss = torch.mean(1 - self.cosine(pred, target)) * 15
        list_channels = list(range(1, pred.shape[1]))
        list_channels.append(0)
        # band shuffle
        sq = torch.Tensor(list_channels).type(torch.LongTensor)

        # shuffle real_img
        base = target[:, sq, :, :]
        newtarget = target - base
        # shuffle fake_img
        base = pred[:, sq, :, :]
        new_fake = pred - base
        spectral_loss2 = self.l1(newtarget, new_fake) * 15

        return spatital_loss + spectral_loss + spectral_loss2
    
class L1SAMLoss(nn.Module):

    def __init__(self, alpha=1.0, eps=1e-8):
        super(L1SAMLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        # Compute L1 loss
        l1_loss = F.l1_loss(pred, target)

        # Compute spectral angle mapper
        dot_product = torch.sum(pred * target, dim=1)
        pred_norm = torch.norm(pred, dim=1)
        target_norm = torch.norm(target, dim=1)
        cosine_similarity = dot_product / (pred_norm * target_norm + self.eps)
        cosine_similarity = torch.clamp(cosine_similarity, -1.0 + self.eps, 1.0 - self.eps)
        spectral_angle = torch.acos(cosine_similarity)
        sam_loss = torch.mean(spectral_angle)

        return l1_loss + self.alpha * sam_loss