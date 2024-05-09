import os
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch import device

from typing import Callable, Tuple, Any
from src.model.vpsnet_learned_malisat_radiometric import VPSNetLearnedMalisatRadiometric
from src.model.vpsnet import VPSNet
from src.utils.metrics import MetricCalculator

import numpy as np
from tqdm import tqdm

def training_epoch(
        model: Module, 
        train_loader: DataLoader, 
        optimizer: Optimizer, 
        loss_f: Callable, 
        metrics: MetricCalculator,  
        device: device,
        epoch: int,
        **kwargs
    ) -> Tuple[Any, ...]:
    
    if kwargs.get('metrics_per_stage', False):
        loss_f.alpha_stage = kwargs.get('stages_parameter', 1.0)

    losses = []
    model.train()
    with tqdm(enumerate(train_loader), total=len(train_loader), leave=False) as pbar:
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _, batch in pbar:
        
            # New batch entering
            optimizer.zero_grad(set_to_none=True)
            gt, ms, lms, pan = batch[0], batch[1], batch[2], batch[3]

            # Loading data to device
            if device == torch.device('cuda'):
                gt = gt.cuda(non_blocking=True).float()  # ground truth
                ms = ms.cuda(non_blocking=True).float()  # multi-spectral in low resolution
                lms = lms.cuda(non_blocking=True).float()  # upsampled multi-spectral
                pan = pan.cuda(non_blocking=True).float()  # panchromatic
            else:
                gt = gt.to(device).float()  # ground truth
                ms = ms.to(device).float()  # multi-spectral in low resolution
                lms = lms.to(device).float()  # upsampled multi-spectral
                pan = pan.to(device).float()  # panchromatic

            # with record_function('forward_pass'):
                # Computing the prediction
            if isinstance(model, VPSNetLearnedMalisatRadiometric):

                if kwargs.get('metrics_per_stage', False):
                    pred, stages, radiometrics = model.forward(ms, lms, pan)
                    loss = loss_f(pred, stages, radiometrics, gt)
                else:
                    pred, radiometrics = model.forward(ms, lms, pan)
                    loss = loss_f(pred, radiometrics, gt)

            else:
                if kwargs.get('metrics_per_stage', False):
                    pred, stages = model.forward(ms, lms, pan)
                else:
                    pred = model.forward(ms, lms, pan)

                # Computing loss function
                # if isinstance(loss_f, RadiometricMSELoss) or isinstance(loss_f, RadiometricL1Loss):
                #     loss = loss_f(pred, pan, model.get_p_tilde(pan.repeat(1,gt.shape[1], 1, 1)), lms, gt)
                
                if kwargs.get('metrics_per_stage', False):
                    loss = loss_f(pred, stages, gt)
                else:
                    loss = loss_f(pred, gt)

            # Backpropagation
            loss.backward()

            # Update of the optimizer
            optimizer.step()

            # Saving results
            loss = loss.float()
            losses.append(loss.cpu().detach().numpy())
            pbar.set_description(
                f"epoch: {epoch} train loss {np.array(losses).mean():.4f}")

            # Computing metrics
            metrics.update(pred.cpu(), gt.cpu(), lms.cpu())
            
        #prof.export_chrome_trace('./trace_epoch.json')

    return np.array(losses).mean(), metrics.dict


def validating_epoch(
        model: Module,
        val_loader: DataLoader, 
        loss_f: Callable, 
        metrics: MetricCalculator, 
        device: device, 
        epoch: int,
        **kwargs
    ) -> Tuple[Any, ...]:

    # os.mkdir(f"images/{epoch}/")

    if kwargs.get('metrics_per_stage', False):
        loss_f.alpha_stage = kwargs.get('stages_parameter', 1.0)
    
    with torch.no_grad():
        model.eval()
        losses = []
        error = 0
        with tqdm(enumerate(val_loader), total=len(val_loader), leave=False) as pbar:
            for _, batch in pbar:
                # New batch entering
                gt, ms, lms, pan = batch[0], batch[1], batch[2], batch[3]

                # Loading data to device
                if device == torch.device('cuda'):
                    gt = gt.cuda(non_blocking=True).float()  # ground truth
                    ms = ms.cuda(non_blocking=True).float()  # multi-spectral in low resolution
                    lms = lms.cuda(non_blocking=True).float()  # upsampled multi-spectral
                    pan = pan.cuda(non_blocking=True).float()  # panchromatic
                else:
                    gt = gt.to(device).float()  # ground truth
                    ms = ms.to(device).float()  # multi-spectral in low resolution
                    lms = lms.to(device).float()  # upsampled multi-spectral
                    pan = pan.to(device).float()  # panchromatic

                # Computing the prediction
                if isinstance(model, VPSNetLearnedMalisatRadiometric):

                    if kwargs.get('metrics_per_stage', False):
                        pred, stages, radiometrics = model.forward(ms, lms, pan)
                        loss = loss_f(pred, stages, radiometrics, gt)
                    else:
                        pred, radiometrics = model.forward(ms, lms, pan)
                        loss = loss_f(pred, radiometrics, gt)

                else:
                    if kwargs.get('metrics_per_stage', False):
                        pred, stages = model.forward(ms, lms, pan)
                    else:
                        pred = model.forward(ms, lms, pan)

                    # Computing loss function
                    # if isinstance(loss_f, RadiometricMSELoss) or isinstance(loss_f, RadiometricL1Loss):
                    #     loss = loss_f(pred, pan, model.get_p_tilde(pan.repeat(1,gt.shape[1], 1, 1)), lms, gt)
                    
                    if kwargs.get('metrics_per_stage', False):
                        loss = loss_f(pred, stages, gt)
                    else:
                        loss = loss_f(pred, gt)

                # Saving results
                loss = loss.float()

                # if isinstance(model, VPSNet) and loss > 0.01 and epoch >= 50 and error <= 5:
                #     imageio.imwrite(f"images/{epoch}/gt_{epoch}_{error}.tif", gt.squeeze(0).cpu().permute((2,1,0)).detach().numpy())
                #     imageio.imwrite(f"images/{epoch}/ms_{epoch}_{error}.tif", ms.squeeze(0).cpu().permute((2,1,0)).detach().numpy())
                #     imageio.imwrite(f"images/{epoch}/lms_{epoch}_{error}.tif", lms.squeeze(0).cpu().permute((2,1,0)).detach().numpy())
                #     imageio.imwrite(f"images/{epoch}/pan_{epoch}_{error}.tif", pan.squeeze(0).cpu().permute((2,1,0)).detach().numpy())
                #     imageio.imwrite(f"images/{epoch}/result_{epoch}_{error}.tif", pred.squeeze(0).cpu().permute((2,1,0)).detach().numpy())

                #     error += 1

                losses.append(loss.cpu().detach().numpy())
                pbar.set_description(
                    f"epoch: {epoch} validation loss {np.array(losses).mean():.4f}")

                # Computing metrics
                metrics.update(pred.cpu(), gt.cpu(), lms.cpu())

    return np.array(losses).mean(), metrics.dict
