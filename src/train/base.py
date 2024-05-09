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
import imageio.v3 as imageio

from src.dataset import dict_datasets
from src.model import dict_model
from src.model.vpsnet import VPSNet
from src.model.vpsnet_post import VPSNetPost
from src.model.vpsnet_learned import VPSNetLearned
from src.utils import dict_optimizers, dict_losses, dict_upsamplings, dict_histograms, default_hp
from src.train.epochs import training_epoch, validating_epoch
from src.utils.losses import RadiometricPerStage
from src.utils.metrics import MetricCalculator
from src.utils.visualization import TensorboardWriter, FileWriter

torch.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
np.random.seed(2024)

gc.collect()
torch.cuda.empty_cache()


class Experiment:
    def __init__(self, dataset="WorldView-3", model="VPSNet", optimizer="Adam", loss_function="MSE",
                 sampling_factor=4, noise_std=None, epochs=1000, hp=default_hp, **kwargs):

        self.dataset = dict_datasets[dataset]

        try:
            self.model = dict_model[model]
        except KeyError:
            self.model = dict_sota[model]

        self.loss_function = loss_function

        self.model_name = model
        self.optimizer = dict_optimizers[optimizer]
        self.criterion = dict_losses[loss_function]
        self.epochs = epochs
        self.sampling_factor = sampling_factor
        self.noise_std = noise_std
        self.hp = hp
        self.kwargs = kwargs
        device_s = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.kwargs["device"] = torch.device(device_s)

        self.kwargs["radiometric"] = isinstance(self.criterion, RadiometricPerStage)
        if self.kwargs.get("histogram", None) is not None:
            self.kwargs["histogram"] = dict_histograms.get(kwargs["histogram"], None)
        self.kwargs["kernel_size"] = self.hp["kernel_size"]
        self.kwargs["std"] = self.hp["std"]

        self.eval_n = max(
            int(epochs * (self.kwargs["evaluation_frequency"] / 100)), 1)
        seed = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self):
        # training and validation data loaders
        dataset_train = self.dataset(self.kwargs["dataset_path"], 'train', noise_level = self.noise_std)
        dataset_val = self.dataset(self.kwargs["dataset_path"], 'validation', noise_level = self.noise_std)


        if self.kwargs.get("upsampling_type", None) is not None:
            memory = 4 if isinstance(self.model, dict_model['VPSNetMemory']) else 0
            upsampling = dict_upsamplings[self.kwargs["upsampling_type"]][0](dataset_train.get_n_channels()+memory, self.sampling_factor)
            downsampling = dict_upsamplings[self.kwargs["upsampling_type"]][1](dataset_train.get_n_channels()+memory, self.sampling_factor)
            self.kwargs["upsamplings"] = [upsampling, downsampling]

        print(f"Batch size: {self.kwargs['batch_size']}")

        if self.kwargs.get("limit_dataset") is None:
            train_loader = DataLoader(
                dataset_train, batch_size=self.kwargs['batch_size'], shuffle=True, num_workers=self.kwargs["num_workers"], pin_memory=True)
            val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True,
                                num_workers=self.kwargs["num_workers"], pin_memory=True)
        else:
            train_loader = self.limit_dataset(dataset_train)
            val_loader = self.limit_dataset(dataset_val, validation=True)

        

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
        
        scheduler_setting = self.kwargs.get("scheduler", None)

        if scheduler_setting == "halved_100":
            scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)

        elif scheduler_setting == "halved_500":
            scheduler = StepLR(self.optimizer, step_size=500, gamma=0.5)

        elif scheduler_setting == "halved_20":
            scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)

        elif scheduler_setting == "almost_same":
            scheduler = StepLR(self.optimizer, step_size=4, gamma=0.99)

        elif scheduler_setting == "halved_200":
            scheduler = StepLR(self.optimizer, step_size=200, gamma=0.5)

        elif scheduler_setting == "halved_only_200":
            scheduler = MultiStepLR(self.optimizer, milestones=[200], gamma=0.5)

        elif scheduler_setting == "halved_only_1400":
            scheduler = MultiStepLR(self.optimizer, milestones=[1400], gamma=0.5)

        else:
            scheduler = None

        # create summary writer if tensorboard_logdir is not None
        writer = TensorboardWriter(val_loader, train_loader, model, self.kwargs["device"], self.kwargs["log_path"], self.kwargs["nickname"], self.noise_std)

        writer.add_text("Model info", self.get_info_model(model, dataset_train), step=None)

        # Initialization of bounding losses
        best_psnr = 0.001

        for epoch in range(start_epoch, self.epochs, 1):
            train_metrics = MetricCalculator(len(dataset_train))
            val_metrics = MetricCalculator(len(dataset_val))
            train_loss, train_metrics = training_epoch(
                model, train_loader, self.optimizer, self.criterion, train_metrics, self.kwargs["device"], epoch, metrics_per_stage = self.kwargs.get('metrics_per_stage', False), stages_parameter = self.kwargs.get('stages_parameter', 1.0))
            val_loss, val_metrics = validating_epoch(
                model, val_loader, self.criterion, val_metrics, self.kwargs["device"], epoch, metrics_per_stage = self.kwargs.get('metrics_per_stage', False), stages_parameter = self.kwargs.get('stages_parameter', 1.0))

            if scheduler is not None:
                scheduler.step()

            if epoch % self.eval_n == 0:
                self._print_data(train_loss, val_loss, epoch, val_metrics)
                if isinstance(model, VPSNet):
                    writer(train_loss, val_loss, None, None, epoch, train_metrics,
                           val_metrics, hyperparameters=model.get_variational_parameters())
                    writer.add_weights_histograms(model, epoch)
                else:
                    writer(train_loss, val_loss, None, None,
                           epoch, train_metrics, val_metrics)

            if (isinstance(model, VPSNet) or isinstance(model, VPSNetPost) or isinstance(model, VPSNetLearned)) and (epoch+1) % self.kwargs.get("stage_step", 2000) == 0 and model.n_iters < self.kwargs.get("stage_max", 10):
                model.n_iters = model.n_iters + self.kwargs.get("stage_inc", 3)
                print(f"Stages: {model.n_iters}")

            psnr = val_metrics["psnr"]

            if val_loss > 1 and epoch >= 100:
                pass

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
        csv_name = self.kwargs.get("csv_name", "./sota.csv")
        csv_split = csv_name.split('.')
        csv_std_name = f".{csv_split[-2]}_std.csv" # Este parche es un poco feo y probablemente no se adapte a todas las situaciones
        
        # Creation of writers
        writer = FileWriter(self.kwargs, output_path, csv_file=csv_name)
        writer_std = FileWriter(self.kwargs, output_path, csv_file=csv_std_name)


        self.kwargs['device'] = torch.device('cpu')

        print(self.kwargs['device'])

        ckpt = torch.load(
            self.kwargs["model_path"], map_location=torch.device('cpu'))
        # model = self._init_model(self.dataset.get_n_channels())
        model = ckpt["model"]

        try:
            model.device = self.kwargs['device']
        except Exception:
            print('Error')

        dataset = self.dataset(self.kwargs["dataset_path"], 'test', noise_level = self.noise_std)

        img_path = f"{self.kwargs.get('images_path', output_path)}/{dataset.get_name()}/images"

        if not os.path.exists(img_path):
            os.makedirs(img_path)


        if isinstance(model, MMNet_addapted):
            state_dict = model.state_dict()
            device_s = 'cpu'  # if torch.cuda.is_available() else 'cpu'
            new_model = MMNet_addapted(
                num_channels=dataset.get_n_channels(), sampling_factor=dataset.get_sampling_factor(), device=device_s)
            new_model.load_state_dict(state_dict, strict=False)
            model = new_model
            model.to(self.kwargs["device"])
            model.width, model.height = 256, 256
        else:
            model.to(self.kwargs['device'])
        # model.n_iters = 30

        dataset = self.dataset(self.kwargs["dataset_path"], 'test', noise_level = self.noise_std)

        # model_summary = self.get_info_model(model, dataset, mode='eval')
        total_parameters = sum(p.numel() for p in model.parameters())

        images_to_save = dataset.IMAGES_TEST_IDX
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.kwargs["num_workers"], pin_memory=False)

        for param in model.parameters():
            param.requires_grad = False

        metrics = MetricCalculator(len(dataset))

        with torch.no_grad():
            model.eval()
            with tqdm(enumerate(data_loader), total=len(data_loader), leave=False) as pbar:
                for idx, batch in pbar:

                    # steps = []
                    # for i in range(self.kwargs.get('stages')):
                    # model.n_iters = i+1
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

                    # steps.append(out)

                    metrics.update(out.cpu(), gt.cpu())

                    # steps = [step.cpu() for step in steps]
                    # metrics.update_stages(steps, gt.cpu())
                    # figure = TensorboardWriter.plot_stages(metrics.steps_dict, model.n_iters)
                    # figure.savefig(f"plot_{idx}.png")
                    # metrics.clean_steps_dict()

                    if idx in images_to_save:
                        # TODO: Repensar nomenclatura imagenes

                        imageio.imwrite(f'{img_path}/{idx}_gt.png', (self.dataset.show_dataset_image(gt).squeeze(
                        ).cpu().permute((1, 2, 0)).detach().numpy()*255).astype(np.uint8), extension='.png')

                        imageio.imwrite(f'{img_path}/{idx}_{self.model_name}.png', (self.dataset.show_dataset_image((out.cpu()-out.cpu().min())/(
                            out.cpu().max()-out.cpu().min())).squeeze().cpu().permute((1, 2, 0)).detach().numpy()*255).astype(np.uint8), extension='.png')
                        
                        imageio.imwrite(f'{img_path}/{idx}_pan.png', np.repeat((pan.squeeze(
                            0).cpu().permute((1, 2, 0)).detach().numpy()*255).astype(np.uint8), 3, axis=-1))
                        
                        imageio.imwrite(f'{img_path}/{idx}_lms.png', (self.dataset.show_dataset_image(lms).squeeze(
                        ).cpu().permute((1, 2, 0)).detach().numpy()*255).astype(np.uint8), extension='.png')

                    # if self.kwargs.get("metrics_per_stage", False):
                    #     metrics.update(out.cpu(), gt.cpu())
                    #     stages = [step.cpu() for step in stages]
                    #     metrics.update_stages(stages, gt.cpu())
                    #     figure = TensorboardWriter.plot_stages(metrics.steps_dict, model.n_iters)
                    #     figure.savefig(f"plot_{idx}.png")
                    #     metrics.clean_steps_dict()
                    # else:
                    #     metrics.update(out.cpu(), gt.cpu())

        csv_save_dict = dict(**metrics.dict, n_params=total_parameters, name=self.model_name, std=self.noise_std) if self.kwargs["nickname"] is None else dict(
                **metrics.dict, model=self.model_name, nickname=self.kwargs["nickname"], std=self.noise_std)
        csv_std_dict = dict(**metrics.dict_std, nickname = self.kwargs.get('nickname', 'Test'), model=self.model_name)
        writer.save_testing_csv(csv_save_dict, csv_name)
        writer_std.save_testing_csv(csv_std_dict, csv_std_name)

    def _save_model(self, model, version, epoch=None):
        try:
            os.makedirs(self.kwargs["snapshot_path"] +
                        f'/ckpt/{self.model_name}/{self.kwargs["nickname"]}_{self.loss_function}')
        except FileExistsError:
            pass
        save_path = self.kwargs["snapshot_path"] + \
            f'/ckpt/{self.model_name}/{self.kwargs["nickname"]}_{self.loss_function}/weights_{version}.pth'
        ckpt = {'experiment': self, 'model': model, 'epoch': epoch}
        torch.save(ckpt, save_path)

    def _init_model(self, n_channels):

        if self.model == dict_sota["GPPNN"]:
            return self.model(
                ms_channels=n_channels,
                pan_channels=1,
                n_feat=64,
                n_layer=8
            )

        elif self.model == dict_sota["MMNet"]:
            device_s = 'cuda' if torch.cuda.is_available() else 'cpu'
            return self.model(num_channels=n_channels, device=device_s, sampling_factor = self.sampling_factor)

        elif  self.model == dict_sota["LAGConv"] or  self.model == dict_sota["NLRNet"]:
            return self.model(n_channels=n_channels)

        elif self.model == dict_sota["MDCUN"] or self.model == dict_sota["S2DBPN"] or self.model == dict_sota["PanFormer"]:
            return self.model(n_channels=n_channels, sampling_factor=self.sampling_factor)

        else:
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

    @staticmethod
    def _do_patches(data, ps):
        data_shape = data.size
        assert data_shape(2) % ps == 0 and data_shape(3) % ps == 0
        patches = unfold(data, kernel_size=ps, stride=ps)
        patch_num = patches.size(2)
        patches = patches.permute(0, 2, 1).view(
            data_shape(0), -1, data_shape(1), ps, ps)
        return torch.reshape(patches, (data_shape(0) * patch_num, data_shape(1), ps, ps))

    @staticmethod
    def _undo_patches(data, n, w, h, ps):
        patches = data.reshape(n, data.size(0), data.size(1), ps, ps)
        patches = patches.view(n, data.size(
            0), data.size(1) * ps * ps).permute(0, 2, 1)
        return fold(patches, (w, h), kernel_size=ps, stride=ps)

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
