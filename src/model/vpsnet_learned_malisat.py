import torch
import torch.nn as nn
from torchinfo import summary
import torchvision.transforms as T

import numpy as np

from torch import Tensor

from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.extend([os.environ.get('PROJECT_PATH')])

from src.postprocessing import dict_post
from upsampling.bicubic import Downsamp_2_2, UpSamp_2_2, UpSamp_4_2, Downsamp_4_2, Upsampling, Downsampling
from model.proxnet import ProxNet


class VPSNetLearnedMalisat(nn.Module):

    def __init__(self, n_channels: int, sampling_factor: int, **kwargs) -> None:
        super().__init__()

        # Model parameters
        self.n_channels = n_channels
        self.n_iters = kwargs["stages"]
        self.kernel_size = kwargs["kernel_size"]
        self.std = kwargs["std"]

        self.metric_per_stage = kwargs.get("metrics_per_stage", False)

        self.postprocessing_type = kwargs.get("postprocessing_type", None)
        self.use_features = kwargs.get("use_features", False)
        self.spectral_blocks = kwargs.get('spectral_blocks', 1)

        if self.postprocessing_type is not None:
            self.postprocessing = self.init_postprocessing()

        # Inicialization of operators
        self.blurr = DobleConv(self.n_channels, kwargs.get("n_features", 32), self.n_channels)

        if sampling_factor == 4:
            self.upsampling = UpSamp_4_2(self.n_channels, sampling_factor=sampling_factor)
            self.downsampling = Downsamp_4_2(self.n_channels, sampling_factor=sampling_factor)
        elif sampling_factor == 2:
            self.upsampling = UpSamp_2_2(self.n_channels, sampling_factor=sampling_factor)
            self.downsampling = Downsamp_2_2(self.n_channels, sampling_factor=sampling_factor)

        self.downsampling_2 = Downsampling(self.n_channels, 2)
        
        # Definition of operators
        self.DB = self.downsampling_support
        self.BU = self.upsampling_support

        # Definition of ProxNet
        self.n_resblocks = kwargs["resblocks"]
        self.proxnet = ProxNet(self.n_channels, self.n_channels, self.n_resblocks)
                        
        # Variational parameters
        self.gamma = nn.Parameter(torch.Tensor([1]))
        self.tau = nn.Parameter(torch.Tensor([0.05]))
        self.lmb = nn.Parameter(torch.Tensor([50]))
        self.mu = nn.Parameter(torch.Tensor([10]))

        self.to(kwargs["device"])

    def forward(self, ms: Tensor, lms: Tensor, pan: Tensor):

        P = pan.repeat(1, self.n_channels, 1, 1) if pan.shape[-3] == 1 else pan

        P_d2 = self.downsampling_2(pan)

        f = ms

        u_tilde = self.BU(ms, pan, P_d2)

        P_tilde = self.BU(self.DB(P, pan, P_d2), pan, P_d2)

        u = lms.clone()
        p = lms.clone()
        q = self.DB(u, pan, P_d2)

        u_barra = u.clone()
        stages = []
        for _ in range(self.n_iters):

            u_anterior = u.clone()

            p = p + self.gamma*u_barra - self.gamma * \
                self.proxnet(p/self.gamma+u_barra)
            
            q = (q + self.gamma*(self.DB(u_barra, pan, P_d2)-f))/(1 + self.gamma/self.mu)

            u = (u - self.tau*(p + self.BU(q, pan, P_d2) - self.lmb*P_tilde*P*u_tilde)) / \
                (1+self.tau*self.lmb*torch.square(P_tilde))
            
            stages.append(u)

            u_barra = 2*u-u_anterior
        
        result = self.postprocessing(u) if self.postprocessing_type is not None else u

        return (result, stages) if self.metric_per_stage else result

    def get_number_params(self):

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        return params
    
    def init_postprocessing(self) -> nn.Module:
        if self.postprocessing_type == "spatial":
            post_model = dict_post[self.postprocessing_type](self.n_channels, self.n_channels)
        elif self.postprocessing_type == "spectral":
            post_model = dict_post[self.postprocessing_type](self.n_channels, n_features=32, reduction = 4, preprocessing = self.use_features)
        elif self.postprocessing_type == "spectralBlocks":
            post_model = dict_post[self.postprocessing_type](self.n_channels, self.spectral_blocks, n_features=32, reduction = 4, preprocessing = self.use_features)
        elif self.postprocessing_type == "parallel":
            post_model = dict_post[self.postprocessing_type](self.n_channels, n_features=32, reduction = 4, preprocessing = self.use_features)
        
        elif self.postprocessing_type == "series":
            post_model = dict_post[self.postprocessing_type](self.n_channels, n_features=32, reduction = 4, preprocessing = self.use_features)

        return post_model
       
    def downsampling_support(self, u: Tensor, pan_full: Tensor, pan_d2: Tensor) -> Tensor:
        
        u_conv = self.blurr(u)

        return self.downsampling(u_conv)
        
    
    def upsampling_support(self, u: Tensor, pan_full: Tensor, pan_d2: Tensor) -> Tensor:
        u_up = self.upsampling(u, pan_d2, pan_full)
        
        return self.blurr(u_up)

    def get_info_model(self):
        return f"Model: VPSNet Learned\n" \
               f"Number of parameters: {self.get_number_params()} \n" \
               f"Postprocessing: {self.postprocessing_type}\n" \
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
    

class DobleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, BN=False):
        super(DobleConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.BN = BN
        if self.BN:
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        if self.BN:
            x = self.ReLU(self.bn1(self.conv1(x)))
            x = self.ReLU(self.bn2(self.conv2(x)))
        else:
            x = self.ReLU(self.conv1(x))
            x = self.ReLU(self.conv2(x))
        return x
    
class FixedConv(nn.Module):

    def __init__(self, kernel_size, padding, stride=1) -> None:
        super(FixedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        x_reshaped = x.reshape((x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3]))
        x_conv = self.conv(x_reshaped)
        x_final = x_conv.reshape(x.shape)
        return x_final

if __name__ == '__main__':
    n_channels = 4

    kwargs = {
        "stages": 3,
        "kernel_size": 9,
        "std": 1.7,
        "resblocks": 1,
        "device": torch.device('cpu'),
        "postprocessing_type": 'spectral',
        "spectral_blocks": 4,
        "use_features": False
        }
    
    model = VPSNetLearnedMalisat(n_channels, 2, **kwargs)

    ms = torch.rand(1, n_channels, 32, 32)
    lms = torch.rand(1, n_channels, 64,64)
    pan = torch.rand(1, 1, 64, 64)

    summary(model, input_data=[ms, lms, pan], mode='train', depth=2, device=kwargs["device"])