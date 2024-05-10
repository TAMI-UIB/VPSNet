import gc
import math
import os
import sys
from typing import Literal, Tuple

import numpy as np
import torch
from torch.nn.functional import fold, unfold
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchinfo import summary
from tqdm import tqdm
import imageio.v3 as imageio

from src.dataset import dict_datasets
from src.model import dict_model
from src.model.vpsnet import VPSNet
from src.model.vpsnet_post import VPSNetPost
from src.model.vpsnet_learned import VPSNetLearned
from src.utils import dict_optimizers, dict_losses, default_hp
from src.train.epochs import training_epoch, validating_epoch
from src.utils.losses import RadiometricPerStage
from src.utils.metrics import MetricCalculator
from src.utils.visualization import TensorboardWriter, FileWriter

class Experiment:
    def __init__(self, dataset="worldview", model="VPSNet", optimizer="Adam", loss_function="L1",
                 sampling_factor=4, noise_std=None, epochs=1000, hp=default_hp, **kwargs):

        self.dataset = dict_datasets[dataset]

        self.model = dict_model[model]

        self.loss_function = loss_function

        self.model_name = model
        self.optimizer = dict_optimizers[optimizer]
        self.criterion = dict_losses[loss_function]
        self.epochs = epochs
        self.sampling_factor = sampling_factor
        self.noise_std = noise_std
        self.hp = hp
        self.kwargs = kwargs

        self.kwargs["radiometric"] = isinstance(self.criterion, RadiometricPerStage)
       
        self.kwargs["kernel_size"] = self.hp["kernel_size"]
        self.kwargs["std"] = self.hp["std"]

        self.eval_n = max(int(epochs * (self.kwargs["evaluation_frequency"] / 100)), 1)
        seed = 2024
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)

    def train(self):
        # training and validation data loaders
        dataset_train = self.dataset(self.kwargs["dataset_path"], 'train', noise_level = self.noise_std)
        dataset_val = self.dataset(self.kwargs["dataset_path"], 'validation', noise_level = self.noise_std)

        print(f"Batch size: {self.kwargs['batch_size']}")

        # Limiting the size of the dataset
        if self.kwargs.get("limit_dataset") is None:
            train_loader = DataLoader(dataset_train, batch_size=self.kwargs['batch_size'], shuffle=True, num_workers=self.kwargs["num_workers"], pin_memory=True)
            val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=self.kwargs["num_workers"], pin_memory=True)
        else:
            train_loader = self.limit_dataset(dataset_train)
            val_loader = self.limit_dataset(dataset_val, validation=True)   

        # Resuming from checkpoint if it is indicated
        if self.kwargs["resume_path"] is not None:

            ckpt = torch.load(
                self.kwargs["resume_path"], map_location=torch.device(self.kwargs["device"]))
            model = ckpt['model']
            
            if 'epoch' in ckpt.keys():
                start_epoch = ckpt['epoch']+1
            else:
                start_epoch = 0

            self.optimizer = ckpt["experiment"].optimizer

        else:
            model = self._init_model(dataset_train.get_n_channels())
            start_epoch = 0
            self.optimizer = self.optimizer(
            model.parameters(), lr=self.hp.get('learning_rate', 1e-5))

        model = model.float()
        model.to(self.kwargs["device"])

        print(f"Devices:{next(model.parameters()).device}")
        
        scheduler = StepLR(self.optimizer, step_size=500, gamma=0.5)

        # create summary writer if tensorboard_logdir is not None
        writer = TensorboardWriter(val_loader, train_loader, model, self.kwargs["device"], self.kwargs["log_path"], self.kwargs["nickname"], self.noise_std)

        writer.add_text("Model info", self.get_info_model(model, dataset_train), step=None)

        best_psnr = 0.001

        for epoch in range(start_epoch, self.epochs, 1):

            # Initializing metrics tracker
            dataset_length_train = self.kwargs.get("limit_dataset", len(dataset_train))
            dataset_length_validation = math.ceil(self.kwargs.get("limit_dataset")/8) if self.kwargs.get("limit_dataset", None) is not None else len(dataset_val)

            train_metrics = MetricCalculator(dataset_length_train)
            val_metrics = MetricCalculator(dataset_length_validation)

            # Training epoch
            train_loss, train_metrics = training_epoch(
                model, train_loader, self.optimizer, self.criterion, train_metrics, self.kwargs["device"], epoch, metrics_per_stage = self.kwargs.get('metrics_per_stage', False), stages_parameter = self.kwargs.get('stages_parameter', 1.0))
            
            # Validation epoch
            val_loss, val_metrics = validating_epoch(
                model, val_loader, self.criterion, val_metrics, self.kwargs["device"], epoch, metrics_per_stage = self.kwargs.get('metrics_per_stage', False), stages_parameter = self.kwargs.get('stages_parameter', 1.0))
            
            scheduler.step()

            # Printing epoch evaluation metrics
            if epoch % self.eval_n == 0:
                self._print_data(train_loss, val_loss, epoch, val_metrics)
                if isinstance(model, VPSNet):
                    writer(train_loss, val_loss, None, None, epoch, train_metrics,
                           val_metrics, hyperparameters=model.get_variational_parameters())
                    writer.add_weights_histograms(model, epoch)
                else:
                    writer(train_loss, val_loss, None, None,
                           epoch, train_metrics, val_metrics)

            psnr = val_metrics["psnr"]

            if psnr >= best_psnr:
                best_psnr = psnr
                self._save_model(model, 'best', epoch)
                writer(train_loss, val_loss, None, None,
                       epoch, train_metrics, val_metrics)
                writer.add_text("best metrics epoch in validation",
                                str(val_metrics), epoch)

            self._save_model(model, 'last', epoch)

        writer.close()

    def eval(self, output_path):
        # Definition of metrics file name
        csv_name = self.kwargs.get("csv_name", "./vpsnet.csv")
        csv_split = csv_name.split('.')
        csv_std_name = f".{csv_split[-2]}_std.csv" 
        
        # Creation of writers
        writer = FileWriter(self.kwargs, output_path, csv_file=csv_name)
        writer_std = FileWriter(self.kwargs, output_path, csv_file=csv_std_name)

        print(self.kwargs['device'])

        ckpt = torch.load(self.kwargs["model_path"], map_location=torch.device('cpu'))
        model = ckpt["model"]

        try:
            model.device = self.kwargs['device']
        except Exception:
            print('Error')

        dataset = self.dataset(self.kwargs["dataset_path"], 'test', noise_level = self.noise_std)

        img_path = f"{self.kwargs.get('images_path', output_path)}/{dataset.get_name()}/images"

        if not os.path.exists(img_path):
            os.makedirs(img_path)

        model.to(self.kwargs['device'])

        dataset = self.dataset(self.kwargs["dataset_path"], 'test', noise_level = self.noise_std)

        self.get_info_model(model, dataset, mode='eval')
        total_parameters = sum(p.numel() for p in model.parameters())

        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.kwargs["num_workers"], pin_memory=False)

        for param in model.parameters():
            param.requires_grad = False

        metrics = MetricCalculator(len(dataset))

        with torch.no_grad():
            model.eval()
            with tqdm(enumerate(data_loader), total=len(data_loader), leave=False) as pbar:
                for idx, batch in pbar:
                    gt, ms, lms, pan = batch[0], batch[1], batch[2], batch[3]
                    gt = gt.to(self.kwargs["device"]).float()
                    ms = ms.to(self.kwargs["device"]).float()
                    lms = lms.to(self.kwargs["device"]).float()
                    pan = pan.to(self.kwargs["device"]).float()

                    out = model.forward(ms, lms, pan)

                    if len(out) == 2:
                        out, _ = out
                    elif len(out) == 3:
                        out, _, _ = out

                    out = out.clamp(min=0, max=1)

                    metrics.update(out.cpu(), gt.cpu())

        csv_save_dict = dict(**metrics.dict, model=self.model_name, nickname=self.kwargs("nickname", "Evaluation"), parameters=total_parameters, noise_std=self.noise_std)
        csv_std_dict = dict(**metrics.dict_std, nickname = self.kwargs.get('nickname', 'Evaluation'), parameters=total_parameters, model=self.model_name)

        writer.save_testing_csv(csv_save_dict, csv_name)
        writer_std.save_testing_csv(csv_std_dict, csv_std_name)

    def _save_model(self, model, version, epoch=None):

        try:
            os.makedirs(self.kwargs["snapshot_path"] + f'/ckpt/{self.model_name}/{self.kwargs["nickname"]}_{self.loss_function}')
        except FileExistsError:
            pass

        save_path = self.kwargs["snapshot_path"] + f'/ckpt/{self.model_name}/{self.kwargs["nickname"]}_{self.loss_function}/weights_{version}.pth'
        ckpt = {'experiment': self, 'model': model, 'epoch': epoch}
        torch.save(ckpt, save_path)

    def _init_model(self, n_channels):
        return self.model(
                n_channels=n_channels,
                sampling_factor=self.sampling_factor,
                **self.kwargs
            )

    def get_info_model(self, model, dataset, mode = 'train'):
        n_channels = dataset.get_n_channels()
        ms_input = torch.rand(1, n_channels, 64 // self.sampling_factor, 64 // self.sampling_factor)
        lms_input = torch.rand(1, n_channels, 64, 64)
        pan_input = torch.rand(1, 1, 64, 64)
        return str(summary(model, input_data=[ms_input, lms_input, pan_input], mode=mode, device=self.kwargs["device"]))

    @staticmethod
    def _print_data(train_loss, val_loss, epoch, metrics):
        metrics_message = ", ".join(
            [f"{k} {v:.2f}" for k, v in metrics.items()])
        print(
            f"epoch {epoch}: trainloss {train_loss:.4f}, valloss {val_loss:.4f}, {metrics_message}")

    def load_from_dict(self, **ckpt):
        for k, v in ckpt.items():
            setattr(self, k, v)

    def limit_dataset(self, dataset: Dataset, validation: bool = False) -> DataLoader:

        indices = np.arange(len(dataset))
        np.random.shuffle(indices)

        limit = self.kwargs.get("limit_dataset") if not validation else math.ceil(self.kwargs.get("limit_dataset")/8)

        generator = torch.Generator()
        generator.manual_seed(2024)

        B = self.kwargs['batch_size'] if not validation else 1
        return DataLoader(dataset, batch_size=B, num_workers=self.kwargs["num_workers"], pin_memory=True, sampler=SubsetRandomSampler(indices[:limit], generator=generator))
