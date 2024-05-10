import argparse
import gc
import os
import sys
from datetime import datetime

import torch
import numpy as np
from dotenv import load_dotenv

if not os.environ.get('SKIP_DOTENV'):
    load_dotenv()

sys.path.extend([os.environ.get('PROJECT_PATH')])

from src.train.base import Experiment
from src.dataset import dict_datasets
from src.postprocessing import dict_post
from src.model import dict_model
from src.utils import dict_losses
from src.upsampling import dict_upsamplings
from src.utils import default_hp

torch.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
np.random.seed(2024)

gc.collect()
torch.cuda.empty_cache()

now = datetime.now()
time = now.strftime("%Y-%m-%d")

def get_log_path(experiment_params):
    global time
    return experiment_params['snapshot_path'] + f"/{experiment_params['dataset']}/{time}/{experiment_params['nickname']}_{experiment_params['loss_function']}_{experiment_params['model']}"

def parse_params(args_env):
    for key, value in args_env.items():
        match key:
            case 'num_workers' | 'batch_size' | 'evaluation_frequency' | 'epochs':
                args_env[key] = int(value)
            case 'device':
                args_env[key] = torch.device(value)
            case _:
                args_env[key] = str(value)
    
    return args_env

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            try:
                match key:
                    case 'iter_stages' | 'iter_resnet' | 'memory_feat' | 'patchsize' | 'windows_size' | 'att_feat' | \
                         'resblock_feat':
                        getattr(namespace, self.dest)[key] = int(value)
                    case _:
                        getattr(namespace, self.dest)[key] = float(value)
            except ValueError:
                getattr(namespace, self.dest)[key] = value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion with unfolding")

    # Dataset Options
    parser.add_argument("--dataset", default="worldview", type=str, help="Dataset for training", choices=dict_datasets.keys())
    parser.add_argument("--limit_dataset", type=int, help="Number of samples of the dataset to use")
    parser.add_argument("--sampling_factor", default=4, type=int, help="Downsampling factor")

    # Model to execute
    parser.add_argument("--model", default="VPSNet", type=str, help="Model to train", choices=dict_model.keys())

    # Extras for the model
    parser.add_argument("--memory_size", type=int, help="Nº of channels for memory", default=4)
    parser.add_argument("--upsampling_type", default="bicubic", type=str, help="Up/Downsampling function",  choices=dict_upsamplings.keys())
    parser.add_argument("--postprocessing_type", type=str, default=None, help="Postprocessing applied in VPSNetPost",  choices=dict_post.keys())
    parser.add_argument("--metrics_per_stage", action='store_true')
    
    # Training parameters
    parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer")
    parser.add_argument("--loss_function", default="MSE", type=str, help="Loss function",  choices=dict_losses.keys())
    parser.add_argument("--stages_parameter", default=1.0, type=float, help="Stages parameter for PerStages losses")
    parser.add_argument("--epochs", type=int, help="Nº Epochs", default=os.environ.get('EPOCHS', 1000))
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=os.environ.get('BATCH_SIZE', 5))

    # Name of the experiment
    parser.add_argument("--nickname", type=str, help="Name for the experiment")

    # Resume training    
    parser.add_argument("--resume_path", type=str, help="Resume path")
    
    # Hyperparameters needed
    parser.add_argument("--hp", nargs='*', action=ParseKwargs, help="Hyper parameters", default=default_hp)

    args = parser.parse_args()
    args_dict = args.__dict__
    print(args_dict)

    # Loading environment arguments
    dict_env = dict(os.environ)

    # TODO: Arreglar esto para que no quede tan chapuza
    dict_env = parse_params({k.lower(): v for k, v in dict_env.items()})
    del dict_env["epochs"]
    del dict_env["batch_size"]

    args_experiment = {**args_dict, **dict_env}
    args_experiment["log_path"] = get_log_path(args_experiment)
    args_experiment["snapsot_path"] = get_log_path(args_experiment)
    
    experiment = Experiment(**args_experiment)
    experiment.train()
