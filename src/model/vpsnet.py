import torch
import torch.nn as nn
from torchinfo import summary
import torchvision.transforms as T

import numpy as np

from typing import Tuple, Any
from torch import Tensor

from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.extend([os.environ.get('PROJECT_PATH')])

from upsampling.bicubic import Upsampling, Downsampling, GaussianSmoothing
from model.proxnet import ProxNet
from src.utils.losses import RadiometricLoss

class VPSNet(nn.Module):

    def __init__(self, n_channels: int, sampling_factor: int, **kwargs) -> None:
        super().__init__()

        # Model parameters
        self.n_channels = n_channels
        self.n_iters = kwargs["stages"]
        self.kernel_size = kwargs["kernel_size"]
        self.std = kwargs["std"]
        self.metrics_per_stage = kwargs.get("metrics_per_stage", False)



        # Inicialization of operators
        try:
            self.blurr = kwargs["convolution"] if kwargs["convolution"] is not None else T.Lambda(
            GaussianSmoothing(self.n_channels, self.kernel_size, self.std).to(kwargs["device"]).forward)
        except KeyError:
            self.blurr = T.Lambda(GaussianSmoothing(self.n_channels, self.kernel_size, self.std).to(kwargs["device"]).forward)

        self.upsampling = kwargs["upsamplings"][0]
        self.downsampling = kwargs["upsamplings"][1]

        # Definition of operators
        self.DB = T.Compose([self.blurr, self.downsampling])
        self.BU = T.Compose([self.upsampling, self.blurr])

        # Definition of ProxNet
        self.n_resblocks = kwargs["resblocks"]
        self.proxnet = ProxNet(
            self.n_channels, self.n_channels, self.n_resblocks)
        
        self.radiometric = RadiometricLoss() if kwargs.get("radiometric", False) else None
                
        # Variational parameters
        self.gamma = nn.Parameter(torch.Tensor([1]))
        self.tau = nn.Parameter(torch.Tensor([0.05]))
        self.lmb = nn.Parameter(torch.Tensor([50]))
        self.mu = nn.Parameter(torch.Tensor([10]))

        self.radiometric_per_stage = []

        self.to(kwargs["device"])

    def forward(self, ms: Tensor, lms: Tensor, pan: Tensor):

        P = pan.repeat(1, self.n_channels, 1, 1) if pan.shape[-3] == 1 else pan
        f = ms
        u_tilde = self.BU(ms) 
        P_tilde = self.get_p_tilde(P)

        u = lms.clone()
        p = lms.clone()
        q = self.DB(u)

        u_barra = u.clone()
        stages = []

        for _ in range(self.n_iters):

            u_anterior = u.clone()

            p = p + self.gamma*u_barra - self.gamma * \
                self.proxnet(p/self.gamma+u_barra)
            q = (q + self.gamma*(self.DB(u_barra)-f))/(1 + self.gamma/self.mu)
            u = (u - self.tau*(p + self.BU(q) - self.lmb*P_tilde*P*u_tilde)) / \
                (1+self.tau*self.lmb*torch.square(P_tilde))

            u_barra = 2*u-u_anterior
            
            if self.radiometric is not None:
                self.radiometric_per_stage.append(self.radiometric(u, P, P_tilde, u_tilde))
                
            if self.metrics_per_stage:
                stages.append(u)


        return (u, stages) if self.metrics_per_stage else u

    def get_number_params(self):

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        return params
    
    def get_info_model(self):
        return f"Model: VPSNet\n" \
               f"Number of parameters: {self.get_number_params()} \n" \
               f"Number of stages: {self.n_iters}\n" \
               f"Upsampling type: {self.upsampling.__class__.__name__} \n"

    def get_radiometric_per_stages(self):
        return self.radiometric_per_stage

    def get_p_tilde(self, pan: Tensor):
        p_tilde = self.BU(self.DB(pan))
        return p_tilde
    
    def get_p_tilde_u(self, u_tilde: Tensor) -> Tensor:
        p_tilde_u = torch.mean(u_tilde, dim=-3).unsqueeze(-3)
        return p_tilde_u
    
    def get_variational_parameters(self):
        return {"lmb": self.lmb, "mu": self.mu, "gamma": self.gamma, "tau": self.tau}

if __name__ == '__main__':

    upsampling = Upsampling(8, 4)
    downsampling = Downsampling(8, 4)

    kwargs = {
        "stages": 3,
        "kernel_size": 9,
        "std": 1.7,
        "upsamplings": [upsampling, downsampling],
        "resblocks": 1,
        "device": torch.device('cpu'),
        "postprocessing_type": "spectral",
        "use_features": True
        }
    
    model = VPSNet(8, 4, **kwargs)

    ms = torch.rand(1, 8, 16, 16)
    lms = torch.rand(1, 8, 64,64)
    pan = torch.rand(1, 1, 64, 64)

    summary(model, input_data=[ms, lms, pan], mode='train', device=kwargs["device"])