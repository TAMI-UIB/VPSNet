import os
import csv
from typing import Any, Dict, Optional, Tuple, List

import torch
from PIL import Image
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.model.vpsnet_learned_malisat import VPSNetLearnedMalisat
from src.model.vpsnet_learned_malisat_radiometric import VPSNetLearnedMalisatRadiometric

import numpy as np
import matplotlib.pyplot as plt

class TensorboardWriter:
    def __init__(
            self, 
            val_loader: DataLoader, 
            train_loader: DataLoader, 
            model: torch.nn.Module, 
            device: str, 
            tensorboard_logdir: str | os.PathLike, 
            name: str,
            std_noise: Optional[float] = None  
        ) -> None:
        
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.model = model
        self.device = device
        self.writer = SummaryWriter(log_dir=tensorboard_logdir)
        self.model_name = name
        self.std_noise = std_noise

    def __call__(
            self, 
            train_loss: Tensor, 
            val_loss: Tensor, 
            train_loss_comp: Optional[dict[str, Any]], 
            val_loss_comp: Optional[dict[str, Any]], 
            epoch: int, 
            train_metrics: dict, 
            validation_metrics: dict, 
            hyperparameters: Optional[dict] = None,
            add_figures: bool = True
        ) -> None:
        
        aux_validation_metrics = dict(validation_metrics)
        aux_train_metrics = dict(train_metrics)

        self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, global_step=epoch)

        self.writer.add_scalars("validation metrics", {'psnr': aux_validation_metrics.pop('psnr'),
                                                       'rmse': aux_validation_metrics.pop('rmse')},
                                global_step=epoch)
        
        self.writer.add_scalars("train metrics", {'psnr': aux_train_metrics.pop('psnr'),
                                                  'rmse': aux_train_metrics.pop('rmse')},
                                global_step=epoch)
        
        aux_validation_metrics['total_loss'] = val_loss

        if val_loss_comp:
            self.writer.add_scalars('validation loss components', val_loss_comp, global_step=epoch)
        
        aux_train_metrics['total_loss'] = train_loss
        
        if train_loss_comp:
            self.writer.add_scalars('train loss components', train_loss_comp, global_step=epoch)
        
        if hyperparameters is not None:
            self.writer.add_scalars('hyperparameters', hyperparameters, global_step=epoch)

        if add_figures:
            figures = self._predict_images()

            for name, fig in figures.items():
                self.writer.add_figure(name, fig, global_step=epoch)

            # gt_v, _, _ = next(iter(self.val_loader))
            # self.writer.add_histogram('Ground Truth', gt_v, epoch)

    @staticmethod           
    def info_stages( metrics: Dict[str, List[float]], n_stages: int, tag: str, epoch: int):
        return TensorboardWriter.plot_stages(metrics, n_stages)
    
    @staticmethod
    def plot_stages(metrics: Dict[str, float], n_stages: int):

        number_of_metrics = len(metrics.keys())

        x = list(range(n_stages))

        fig, axs = plt.subplots(1, number_of_metrics, figsize=(number_of_metrics*3, 3))
        
        for i, metric in enumerate(metrics.keys()):
            # metrics[metric] = [num for num in metrics[metric]]
            axs[i].plot(x, metrics[metric])
            axs[i].set_xlabel("Stages")
            axs[i].set_title(metric)
        
        return fig
        
    def _predict_images(self) -> dict[str, Any]: 
        gt_v, ms_v, lms_v, pan_v = next(iter(self.val_loader))
        gt_t, ms_t, lms_t, pan_t = next(iter(self.train_loader))

        N = gt_v.shape[0]

        # plot at most 5 images even if the batch size is larger
        if N > 5:
            gt_v = gt_v[:5]
            ms_v = ms_v[:5]
            lms_v = lms_v[:5]
            pan_v = pan_v[:5]

            gt_t = gt_t[:5]
            ms_t = ms_t[:5]
            lms_t = lms_t[:5]
            pan_t = pan_t[:5]

        with torch.no_grad():
            if isinstance(self.model, VPSNetLearnedMalisatRadiometric):

                if self.model.metric_per_stage:
                    pred_v, _, _ = self.model.forward(ms_v.to(self.device).float(), lms_v.to(self.device).float(), pan_v.to(self.device).float())
                    pred_t, _, _ = self.model.forward(ms_t.to(self.device).float(), lms_t.to(self.device).float(), pan_t.to(self.device).float())
                else:
                    pred_v, _ = self.model.forward(ms_v.to(self.device).float(), lms_v.to(self.device).float(), pan_v.to(self.device).float())
                    pred_t, _ = self.model.forward(ms_t.to(self.device).float(), lms_t.to(self.device).float(), pan_t.to(self.device).float())
            elif isinstance(self.model, VPSNetLearnedMalisat):
                if self.model.metric_per_stage:
                    pred_v, _ = self.model.forward(ms_v.to(self.device).float(), lms_v.to(self.device).float(), pan_v.to(self.device).float())
                    pred_t, _ = self.model.forward(ms_t.to(self.device).float(), lms_t.to(self.device).float(), pan_t.to(self.device).float())
                else:
                    pred_v = self.model.forward(ms_v.to(self.device).float(), lms_v.to(self.device).float(), pan_v.to(self.device).float())
                    pred_t = self.model.forward(ms_t.to(self.device).float(), lms_t.to(self.device).float(), pan_t.to(self.device).float())
            else:
                pred_v = self.model.forward(ms_v.to(self.device).float(), lms_v.to(self.device).float(), pan_v.to(self.device).float())
                pred_t = self.model.forward(ms_t.to(self.device).float(), lms_t.to(self.device).float(), pan_t.to(self.device).float())
                
        
            gt_v = self.val_loader.dataset.show_dataset_image(gt_v)
            ms_v = self.val_loader.dataset.show_dataset_image(ms_v)
            lms_v = self.val_loader.dataset.show_dataset_image(lms_v)

            pred_v = self.val_loader.dataset.show_dataset_image(pred_v)

            gt_t = self.train_loader.dataset.show_dataset_image(gt_t)
            ms_t = self.train_loader.dataset.show_dataset_image(ms_t)
            lms_t = self.train_loader.dataset.show_dataset_image(lms_t)

            pred_t = self.train_loader.dataset.show_dataset_image(pred_t)

            return dict(
                predictions_val=self._plot_batch(gt_v.to('cpu'), pred_v.to('cpu'), lms_v.to('cpu'), pan_v.to('cpu')),
                predictions_train=self._plot_batch(gt_t.to('cpu'), pred_t.to('cpu'), lms_t.to('cpu'), pan_t.to('cpu'))
            )

    def _plot_batch(self, gt: Tensor, pred: Tensor | Tuple, lms: Tensor, pan: Tensor):
        # xout = xout.detach().cpu().numpy()
        N = min(gt.shape[0], 5)

        n_plots = 4
        height = 3
        width = 3
        fig, axs = plt.subplots(N, n_plots, figsize=((n_plots+1) * width, N * height))

        if N == 1:
            axs = [axs]
            
    
        for axs_row, gt_i, pred_i, lms_i, pan_i in zip(axs, gt, pred, lms, pan):
            self._add_image(axs_row[0], gt_i.clip(0,1), "Ground Truth")
            self._add_image(axs_row[1], pred_i.clip(0,1), "Result")
            self._add_image(axs_row[2], lms_i.clip(0, 1), "Multi-spectral")
            self._add_image(axs_row[3], pan_i.clip(0,1), "PAN")

            [ax.axis("off") for ax in axs_row]
        return fig
    
    def _add_image(self, axs_row: Any, image: Tensor, name: str):
        if len(image.size()) != 3:
            return
        
        image = torch.permute(image, (1, 2, 0))
        axs_row.imshow(image.detach().numpy())
        axs_row.set_title(name)

    def add_weights_histograms(self, model, epoch):
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.writer.add_histogram(name, param, global_step=epoch) 

    def add_text(self, title, content, step):
        self.writer.add_text(title, content, step)

    def close(self):
        self.writer.close()

class FileWriter:

    def __init__(
            self,
            snapshot_path: str | os.PathLike,
            output_path: Optional[str | os.PathLike],
            csv_file: Optional[str | os.PathLike]
        ) -> None:
        
        self.snapshot_path = snapshot_path
        self.output_path = self._get_output_path(output_path)
        self.csv_file = csv_file
    
    def _get_output_path(self, output_path: Optional[str | os.PathLike]) -> str:

        if output_path is None:
            try:
                os.makedirs(self.snapshot_path + '/results')
            except FileExistsError:
                pass
            finally:
                return self.snapshot_path + '/results'
        else:
            try:
                os.makedirs(self.output_path)
            except FileExistsError:
                pass
            finally:
                return output_path

    def save_image(self, image: Tensor, file_name: str) -> None:
        save_image(image, f"{self.output_path}/{file_name}.png")

    def save_testing_csv(self, test_results: dict[str, Any], file_name: Optional[str] = None) -> None:
        for k, v in test_results.items():
            if not isinstance(v, str):
                test_results[k] = np.array(v).mean() if v is not None else v

        file_path = self.csv_file if self.csv_file is not None else f"{self.output_path}/{file_name}_test.csv"
        already_exists = os.path.exists(file_path)
    
        with open(file_path, mode='a', newline='') as f:
            w = csv.DictWriter(f, test_results.keys())

            if not already_exists or self.csv_file is None:
                w.writeheader() 

            w.writerow(test_results)
    
    def save_metrics_csv(self, metrics: dict[str, Any], file_name: str) -> None:

        for k, v in metrics.items():
            metrics[k] = np.array(v).mean()

        if self.csv_file is not None:
            with open(self.csv_file, mode='a', newline='') as f:
                w = csv.DictWriter(f, metrics.keys())

                if not already_exists:
                    w.writeheader()

                w.writerow(metrics)
        else:

            already_exists = os.path.exists(f"{self.output_path}/{file_name}_metrics.csv")

            with open(f"{self.output_path}/{file_name}_metrics.csv", mode='a', newline='') as f:
                w = csv.DictWriter(f, metrics.keys())

                if not already_exists:
                    w.writeheader()

                w.writerow(metrics)
    

