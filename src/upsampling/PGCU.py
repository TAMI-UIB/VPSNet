import torch
import torch.nn as nn
import torch.nn.functional as fun
from math import sqrt

class DownSamplingBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, sampling_factor):
        super(DownSamplingBlock, self).__init__()
        self.Conv = nn.Conv2d(in_channel, out_channel, (3,3), 2, 1)
        self.MaxPooling = nn.MaxPool2d(kernel_size=sampling_factor//2)
        
    def forward(self, x):
        out = self.MaxPooling(self.Conv(x))
        return out
    
class PGCU(nn.Module):
    
    def __init__(self, sampling_factor=4, channel=4, vecLen=128, numberBlocks=3):
        super(PGCU, self).__init__()
        self.BandVecLen = vecLen//channel
        self.channel = channel
        self.vecLen = vecLen
        self.sampling_factor = sampling_factor
        
        ## Information Extraction
        # F.size == (Vec, W, H)
        self.FPConv = nn.Conv2d(1, channel, (3,3), 1, 1)
        self.FMConv = nn.Conv2d(channel, channel, (3,3), 1, 1)
        self.FConv = nn.Conv2d(channel*2, vecLen, (3,3), 1, 1)
        # G.size == (Vec, W/pow(2, N), H/pow(2, N))
        self.GPConv = nn.Sequential()
        self.GMConv = nn.Sequential()
        self.GConv = nn.Conv2d(channel*2, vecLen, (3,3), 1, 1)
        for i in range(numberBlocks):
            if i == 0:
                self.GPConv.add_module('DSBlock'+str(i), DownSamplingBlock(1, channel, sampling_factor=self.sampling_factor))
            else:
                self.GPConv.add_module('DSBlock'+str(i), DownSamplingBlock(channel, channel, sampling_factor=self.sampling_factor))
                self.GMConv.add_module('DSBlock'+str(i-1), DownSamplingBlock(channel, channel, sampling_factor=self.sampling_factor))
        # V.size == (C, W/pow(2, N), H/pow(2, N)), k=W*H/64
        self.VPConv = nn.Sequential()
        self.VMConv = nn.Sequential()
        self.VConv = nn.Conv2d(channel*2, channel, (3,3), 1, 1)
        for i in range(numberBlocks):
            if i == 0:
                self.VPConv.add_module('DSBlock'+str(i), DownSamplingBlock(1, channel, sampling_factor=self.sampling_factor))
            else:
                self.VPConv.add_module('DSBlock'+str(i), DownSamplingBlock(channel, channel, sampling_factor=self.sampling_factor))
                self.VMConv.add_module('DSBlock'+str(i-1), DownSamplingBlock(channel, channel, sampling_factor=self.sampling_factor))

        # Linear Projection
        self.FLinear = nn.ModuleList([nn.Sequential(nn.Linear(self.vecLen, self.BandVecLen), nn.LayerNorm(self.BandVecLen)) for i in range(self.channel)])
        self.GLinear = nn.ModuleList([nn.Sequential(nn.Linear(self.vecLen, self.BandVecLen), nn.LayerNorm(self.BandVecLen)) for i in range(self.channel)])
        # FineAdjust
        self.FineAdjust = nn.Conv2d(channel, channel, (3,3), 1, 1)
        
    def forward(self, x, guide):
        up_x = fun.interpolate(x, scale_factor=(self.sampling_factor,self.sampling_factor), mode='nearest')
        Fm = self.FMConv(up_x)
        Fq = self.FPConv(guide)
        F = self.FConv(torch.cat([Fm, Fq], dim=1))
        
        Gm = self.GMConv(x)
        Gp = self.GPConv(guide)
        G = self.GConv(torch.cat([Gm, Gp], dim=1))
        
        Vm = self.VMConv(x)
        Vp = self.VPConv(guide)
        V = self.VConv(torch.cat([Vm, Vp], dim=1))
        
        C = V.shape[1]
        batch = G.shape[0]
        W, H = F.shape[2], F.shape[3]
        OW, OH = G.shape[2], G.shape[3]
        
        G = torch.transpose(torch.transpose(G, 1, 2), 2, 3)
        G = G.reshape(batch*OW*OH, self.vecLen)
        
        F = torch.transpose(torch.transpose(F, 1, 2), 2, 3)
        F = F.reshape(batch*W*H, self.vecLen)
        BandsProbability = None
        for i in range(C):
            # F projection
            FVF = self.GLinear[i](G)
            FVF = FVF.reshape(batch, OW*OH, self.BandVecLen).transpose(-1, -2) # (batch, L, OW*OH)
            # G projection
            PVF = self.FLinear[i](F)
            PVF = PVF.view(batch, W*H, self.BandVecLen) # (batch, W*H, L)
            # Probability
            Probability = torch.bmm(PVF, FVF).reshape(batch*H*W, OW, OH) / sqrt(self.BandVecLen)
            Probability = torch.exp(Probability) / torch.sum(torch.exp(Probability), dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
            Probability = Probability.view(batch, W, H, 1, OW, OH)
            # Merge
            if BandsProbability is None:
                BandsProbability = Probability
            else:
                BandsProbability = torch.cat([BandsProbability, Probability], dim=3)
        #Information Entropy: H_map = torch.sum(BandsProbability*torch.log2(BandsProbability+1e-9), dim=(-1, -2, -3)) / C
        out = torch.sum(BandsProbability*V.unsqueeze(dim=1).unsqueeze(dim=1), dim=(-1, -2))
        out = out.transpose(-1, -2).transpose(1, 2)
        out = self.FineAdjust(out)
        return out
    