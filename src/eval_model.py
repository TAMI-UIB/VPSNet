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
from src.model import dict_model
from src.utils import dict_upsamplings, dict_histograms


torch.cuda.manual_seed_all(2024)
np.random.seed(2024)

gc.collect()
torch.cuda.empty_cache()

now = datetime.now()
time = now.strftime("%Y-%m-%d")

def get_log_path(experiment_params):
    global time
    return experiment_params['snapshot_path'] + f"/{experiment_params['dataset']}/{time}/{experiment_params['nickname']}_{experiment_params['model']}"

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

class ParseHP(argparse.Action):
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
    parser.add_argument("--dataset", default="worldview", type=str, help="Dataset for evaluation", choices=dict_datasets.keys())
    parser.add_argument("--stages", type=int, help="Nº of stages", default=10)
    parser.add_argument("--resblocks", type=int, help="Nº of resblocks", default=1)
    parser.add_argument("--upsampling_type", default="bicubic", type=str, help="Up/Downsampling function",  choices=dict_upsamplings.keys())
    parser.add_argument("--model", default="VPSNet", type=str, help="Model to evaluate", choices=dict_model.keys())
    parser.add_argument("--nickname", type=str, help="Name for the experiment")
    parser.add_argument("--histogram", type=str, help="Histogram function",  choices=dict_histograms.keys())
    parser.add_argument("--csv_name", default='./ours.csv', type=str, help="Name for the experiment")
    parser.add_argument("--sampling_factor", default=4, type=int, help="Downsampling factor")
    parser.add_argument("--metrics_per_stage", action='store_true')
    parser.add_argument("--model_path", type=str, help="Resume path", required=True)
    parser.add_argument("--noise_std", type=float, help="Standard deviation for noise", default=None)

    # Loading cli arguments
    args = parser.parse_args()
    args_dict = args.__dict__
    print(args_dict)

    # Loading environment arguments
    dict_env = dict(os.environ)
    dict_env = parse_params({k.lower(): v for k, v in dict_env.items()})
    args_experiment = {**args_dict, **dict_env}
    args_experiment["log_path"] = get_log_path(args_experiment)
    args_experiment["device"] = torch.device('cpu')
    experiment = Experiment(**args_experiment)
    experiment.eval(output_path=args_experiment["log_path"])
    