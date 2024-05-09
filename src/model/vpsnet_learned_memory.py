import torch
import torch.nn as nn
from torchinfo import summary
import torchvision.transforms as T

import numpy as np

from torch import Tensor

import sys
sys.path.extend(["/home/marctomas/Escritorio/Repositoris/E2EPansharpening"])

from src.postprocessing import dict_post
from upsampling.upsamplings import UpsamplingConv, DownsamplingConv, Downsampling
from model.proxnet import ProxNet
from src.utils.losses import RadiometricLoss

class VPSNetLearnedMemory(nn.Module):

    def __init__(self, n_channels: int, sampling_factor: int, **kwargs) -> None:
        super().__init__()

        # Model parameters
        self.n_channels = n_channels
        self.n_iters = kwargs["stages"]
        self.kernel_size = kwargs["kernel_size"]
        self.std = kwargs["std"]
        self.memory_size = kwargs.get('memory_size', 4)
        self.metric_per_stage = kwargs.get("metrics_per_stage", False)
        self.use_features = kwargs.get("use_features", False)

        # Postprocessing parsing
        self.postprocessing_type = kwargs.get("postprocessing_type", None)

        if self.postprocessing_type is not None:
            self.postprocessing = self.init_postprocessing()


        # Inicialization of operators
        self.blurr = DobleConv(self.n_channels+self.memory_size, kwargs.get("n_features", 32), self.n_channels+self.memory_size)

        self.upsampling = kwargs["upsamplings"][0]
        self.downsampling = kwargs["upsamplings"][1]

        self.downsampling_2 = Downsampling(self.n_channels, 2)
        
        self.DB = T.Compose([self.blurr, self.downsampling])
        self.BU = T.Compose([self.upsampling, self.blurr])

        # Definition of ProxNet
        self.n_resblocks = kwargs["resblocks"]
        self.proxnet = ProxNet(self.n_channels+self.memory_size, self.n_channels+self.memory_size, self.n_resblocks)
        
        self.radiometric = RadiometricLoss() if kwargs.get("radiometric", False) else None
                
        # Variational parameters
        self.gamma = nn.Parameter(torch.Tensor([1]))
        self.tau = nn.Parameter(torch.Tensor([0.05]))
        self.lmb = nn.Parameter(torch.Tensor([50]))
        self.mu = nn.Parameter(torch.Tensor([10]))

        self.radiometric_per_stage = []
        self.device = kwargs["device"] 
        self.to(self.device)

    def forward(self, ms: Tensor, lms: Tensor, pan: Tensor):

        P = pan.repeat(1, self.n_channels, 1, 1) if pan.shape[-3] == 1 else pan

        f = ms
        f_extended = self.extend_tensor(ms)

        u_tilde = self.BU(f_extended)[:,:self.n_channels,:,:]
        P_tilde = self.get_p_tilde(self.extend_tensor(P))[:,:self.n_channels,:,:]

        u_extended = self.extend_tensor(lms)
        p_extended = self.extend_tensor(lms)
        q_extended = self.DB(u_extended)

        u_barra_extended = u_extended.clone()

        den_u = 1+self.tau*self.lmb*torch.square(P_tilde)
        den_q = (1 + self.gamma/self.mu)
        const_1_u = self.lmb*P_tilde*P*u_tilde

        for _ in range(self.n_iters):

            u_anterior = u_extended.clone()

            p_extended = p_extended + self.gamma*u_barra_extended - self.gamma * self.proxnet(p_extended/self.gamma+u_barra_extended)
            
            q_extended = (q_extended + self.gamma*(self.DB(u_barra_extended)-f_extended)) / den_q
            
            u, u_memory = torch.split(u_extended, self.n_channels, dim=1)
            p = torch.split(p_extended, self.n_channels, dim=1)[0]
            q_up = torch.split(self.BU(q_extended), self.n_channels, dim=1)[0]

            u = (u - self.tau*(p + q_up - const_1_u)) / den_u

            u_extended = torch.concat((u, u_memory), dim = 1)
            u_barra_extended = 2*u_extended-u_anterior
            
            if self.radiometric is not None:
                self.radiometric_per_stage.append(self.radiometric(u_extended[:,:self.n_channels,:,:], P, P_tilde, u_tilde))
            
        result = self.postprocessing(u_extended[:,:self.n_channels,:,:]) if self.postprocessing_type is not None else u_extended[:,:self.n_channels,:,:] 

        return result

    def get_number_params(self):

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        return params
    
    def extend_tensor(self, tensor: Tensor) -> Tensor:
        return torch.concat((tensor.clone(), torch.zeros((tensor.shape[0], self.memory_size, tensor.shape[2], tensor.shape[3]), device=self.device)), dim=1)
    
    def init_postprocessing(self) -> nn.Module:
        if self.postprocessing_type == "spatial":
            post_model = dict_post[self.postprocessing_type](self.n_channels, self.n_channels)
        elif self.postprocessing_type == "spectral":
            post_model = dict_post[self.postprocessing_type](self.n_channels, n_features=32, reduction = 4, preprocessing = self.use_features)
        
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
    n_channels = 8
    memory_size = 8
    upsampling = UpsamplingConv(n_channels+memory_size, 4)
    downsampling = DownsamplingConv(n_channels+memory_size, 4)

    kwargs = {
        "stages": 3,
        "kernel_size": 9,
        "std": 1.7,
        "upsamplings": [upsampling, downsampling],
        "resblocks": 1,
        "device": torch.device('cpu'),
        "postprocessing_type": None,
        "memory_size": memory_size,
        "use_features": True
        }
    
    model = VPSNetLearnedMemory(n_channels, 4, **kwargs)

    ms = torch.rand(1, n_channels, 16, 16)
    lms = torch.rand(1, n_channels, 64,64)
    pan = torch.rand(1, 1, 64, 64)

    summary(model, input_data=[ms, lms, pan], mode='train', device=kwargs["device"])