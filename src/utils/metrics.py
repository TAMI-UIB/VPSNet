import math
import torch
import numpy as np
import torch.nn.functional as f
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM

def ERGAS(pred, target, sampling_factor):
    channel_rmse = torch.mean(torch.sqrt(torch.mean(torch.square(pred - target), dim=(2, 3))))
    channel_mean = torch.mean(pred, dim=(2, 3))
    channel_sum = torch.mean(torch.div(channel_rmse, channel_mean)**2, dim=1)
    return 100 * sampling_factor * torch.mean(torch.sqrt(channel_sum))


def RMSE(pred, target):
    return torch.mean(torch.sqrt(torch.mean(torch.square(pred - target), dim=(1, 2, 3))))


def PSNR(pred, target):
    psnr_list = -10 * torch.log10(torch.mean(torch.square(pred - target), dim=(1, 2, 3)))
    return torch.mean(psnr_list)

def Q_index(pred, target):
    channels, height, width = target.shape[0], target.shape[1], target.shape[2]

    # Ensure the inputs have the same shape
    if target.shape != pred.shape:
        raise ValueError("target and pred images must have the same shape.")

    # Split the images into multiple bands
    target_bands = torch.chunk(target, channels, dim=0)
    pred_bands = torch.chunk(pred, channels, dim=0)

    q_index_values = []

    for i in range(channels):
        # Calculate mean and standard deviation for each band
        mean_ref = target_bands[i].mean()
        mean_dis = pred_bands[i].mean()
        std_ref = target_bands[i].std()
        std_dis = pred_bands[i].std()

        # Calculate cross-covariance for each band
        cov = torch.mean((target_bands[i] - mean_ref) * (pred_bands[i] - mean_dis))

        # Calculate the luminance component
        luminance = 2 * mean_ref * mean_dis / (mean_ref ** 2 + mean_dis ** 2)

        # Calculate the contrast component
        contrast = 2 * std_ref * std_dis / (std_ref ** 2 + std_dis ** 2)

        # Calculate the structure component
        structure = cov / (std_ref * std_dis)

        # Calculate the q_index for the band
        q_index = luminance * contrast * structure

        q_index_values.append(q_index)

    # Calculate the final q_index for the image
    total_q_index = torch.mean(torch.stack(q_index_values))
    
    return total_q_index





def D_lambda(img_fake, img_lm, block_size=32, p=1):
    """Spectral distortion
    img_fake, generated HRMS
    img_lm, LRMS"""
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    # D_lambda
    Q_fake = []
    Q_lm = []
    for i in range(C_f):
        for j in range(i + 1, C_f):
            # for fake
            band1 = img_fake[..., i]
            band2 = img_fake[..., j]
            Q_fake.append(Q_index(band1, band2, block_size=block_size))
            # for real
            band1 = img_lm[..., i]
            band2 = img_lm[..., j]
            Q_lm.append(Q_index(band1, band2, block_size=block_size))
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    D_lambda_index = (np.abs(Q_fake - Q_lm) ** p).mean()
    return D_lambda_index ** (1 / p)


def D_s(img_fake, img_lm, pan, scale=4, block_size=32, q=1):
    """Spatial distortion
    img_fake, generated HRMS
    img_lm, LRMS
    pan, HRPan"""

    # fake and lm
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape

    # fake and pan

    H_p, W_p, C_p = pan.shape

    # get LRPan, 2D
    pan_lr = f.interpolate(pan, scale_factor=1/scale, mode='bicubic') # Esto deberia ser P_tilde (falta convolucionar antes)

    # D_s
    Q_hr = []
    Q_lr = []
    for i in range(C_f):

        # for HR fake
        band1 = img_fake[..., i]
        band2 = pan[..., 0]  # the input PAN is 3D with size=1 along 3rd dim

        Q_hr.append(Q_index(band1, band2, block_size=block_size))
        band1 = img_lm[..., i]
        band2 = pan_lr  # this is 2D

        Q_lr.append(Q_index(band1, band2, block_size=block_size))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    D_s_index = (np.abs(Q_hr - Q_lr) ** q).mean()
    return D_s_index ** (1 / q)


def qnr(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, p=1, q=1, alpha=1, beta=1):
    """QNR - No reference IQA"""
    D_lambda_idx = D_lambda(img_fake, img_lm, block_size, p)
    D_s_idx = D_s(img_fake, img_lm, pan, scale, block_size, q)
    QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
    return QNR_idx

def SCC(pred, target):
    eje = 1
    if pred.shape[-2] == 1 and target.shape[-2] == 1:
        eje = 1
        return torch.corrcoef(torch.cat([pred, target], axis=eje))[0, 1].item()
    else:
        correlations = [torch.corrcoef(torch.cat([pred[i], target[i]], axis=eje)).item() for i in range(pred.shape[-2])]
        return np.mean(correlations)


def SAM(pred, target):
    scalar_dot = torch.sum(torch.mul(pred, target), dim=(1, 2, 3), keepdim=True)
    norm_pred = torch.sqrt(torch.sum(pred**2, dim=(1, 2, 3), keepdim=True))
    norm_target = torch.sqrt(torch.sum(target**2, dim=(1, 2, 3), keepdim=True))
    return torch.mean(torch.arccos(scalar_dot/(norm_pred*norm_target)))


class MetricCalculator:
    def __init__(self, dataset_len, sampling_factor=4):

        self.len = dataset_len
        self.steps_dict = {'ergas': [], 'rmse': [], 'psnr': [], 'ssim': [], 'sam': []}
        self.dict = {'ergas': 0, 'rmse': 0, 'psnr': 0, 'psnr_inter_pred':0, 'ssim': 0, 'sam': 0, 'psnr_inter_gt': 0, 'q-index': 0, 'scc': 0, 'd_s': 0, 'd_lmb': 0, "qnr": 0}
        self.dict_s1 = {'ergas': 0, 'rmse': 0, 'psnr': 0, 'psnr_inter_pred':0, 'ssim': 0, 'sam': 0, 'psnr_inter_gt': 0, 'q-index': 0, 'scc': 0, 'd_s': 0, 'd_lmb': 0, "qnr": 0}
        self.dict_s2 = {'ergas': 0, 'rmse': 0, 'psnr': 0, 'psnr_inter_pred':0, 'ssim': 0, 'sam': 0, 'psnr_inter_gt': 0, 'q-index': 0, 'scc': 0, 'd_s': 0, 'd_lmb': 0, "qnr": 0}
        self.dict_std = {'ergas': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0, 'sam': 0, 'q-index': 0, 'scc': 0, 'd_s': 0, 'd_lmb': 0, "qnr": 0}
        self.sampling_factor = sampling_factor

    def update(self, pred, target, inter = None):

        rmse = RMSE(pred, target).item()
        q_index = Q_index(pred, target).item()
        psnr = PSNR(pred, target).item()
        ergas = ERGAS(pred, target, self.sampling_factor).item()
        ssim = SSIM(pred, target, data_range=1.).item()
        sam = SAM(pred, target).item()
        # scc = SCC(pred, target)

        N = pred.shape[0]

        # Computation of the mean for every metric
        self.dict['sam'] += N * sam / self.len
        self.dict['q-index'] += N * q_index / self.len
        self.dict['ergas'] += N * ergas / self.len
        self.dict['rmse'] += N * rmse / self.len
        self.dict['psnr'] += N * psnr / self.len
        self.dict['ssim'] += N * ssim / self.len

        # Computation of std for every metric
        self.dict_s1['sam'] += N * sam 
        self.dict_s1['q-index'] += N * q_index 
        self.dict_s1['ergas'] += N * ergas 
        self.dict_s1['rmse'] += N * rmse 
        self.dict_s1['psnr'] += N * psnr 
        self.dict_s1['ssim'] += N * ssim

        self.dict_s2['sam'] += N * sam**2 
        self.dict_s2['q-index'] += N * q_index**2 
        self.dict_s2['ergas'] += N * ergas**2
        self.dict_s2['rmse'] += N * rmse**2
        self.dict_s2['psnr'] += N * psnr**2
        self.dict_s2['ssim'] += N * ssim**2

        self.dict_std = {k: math.sqrt(self.len*self.dict_s2[k] - (self.dict_s1[k])**2)/self.len for k in self.dict_std.keys()}

        if inter is not None:
            psnr_inter_pred = PSNR(pred, inter).item()
            psnr_inter_gt = PSNR(target, inter).item()
            self.dict['psnr_inter_pred'] += N * psnr_inter_pred / self.len
            self.dict['psnr_inter_gt'] += N * psnr_inter_gt / self.len

    def clean_steps_dict(self):
        self.steps_dict = {'ergas': [], 'rmse': [], 'psnr': [], 'ssim': [], 'sam': []}
        
    def update_stages(self, steps, target):
        
        for step in steps:
            self.steps_dict['sam'].append(SAM(step, target))
            self.steps_dict['ergas'].append(ERGAS(step, target, self.sampling_factor)) 
            self.steps_dict['rmse'].append(RMSE(step, target)) 
            self.steps_dict['psnr'].append(PSNR(step, target)) 
            self.steps_dict['ssim'].append(SSIM(step, target)) 