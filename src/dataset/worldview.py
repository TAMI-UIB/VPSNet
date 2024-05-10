import torch
from h5py import File
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
import sys
import dotenv
dotenv.load_dotenv()
sys.path.extend([os.environ.get('PROJECT_PATH')])


from upsampling.bicubic import Upsampling, Downsampling, GaussianSmoothing

from typing import Literal, Tuple

class WorldView3Dataset(Dataset):

    IMAGES_TEST_IDX = [0, 1, 2, 4, 5, 7, 10, 11, 14, 16]

    def __init__(self, path_to_dataset: str, dataset_mode: Literal['train', 'validation', 'test'], generated_from_gt: bool = True, **kwargs) -> None:
        super().__init__()

        
        match dataset_mode:
            case 'train':
                self.path = path_to_dataset + "/WorldView-3/train/train_wv3.h5"

            case 'validation':
                self.path = path_to_dataset + "/WorldView-3/train/validation_wv3.h5"

            case 'test':
                self.path = path_to_dataset + "/WorldView-3/test/test_wv3_multiExm1.h5"
                
        self.dataset_mode = dataset_mode
        self.dataset_file = File(self.path)
        self.generated_from_gt = generated_from_gt

        if self.generated_from_gt:
            self.upsampling = Upsampling(8, 4)
            self.downsampling = Downsampling(8, 4)
            self.conv = GaussianSmoothing(8, 9, 1.7)

    def __len__(self) -> int:
        return self.dataset_file['pan'].shape[0]
    
    @staticmethod
    def get_name():
        return 'worldview'

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        gt = (torch.from_numpy(self.dataset_file['gt'][idx]) / 2047).clamp(0,1).float()
        
        if self.generated_from_gt:
            pan, ms, lms = self.generate_data_from_gt(gt)
        else:
            ms = (torch.from_numpy(self.dataset_file['ms'][idx]) / 2047).clamp(0,1)
            lms = (torch.from_numpy(self.dataset_file['lms'][idx]) / 2047).clamp(0,1)
            pan = (torch.from_numpy(self.dataset_file['pan'][idx]) / 2047).clamp(0,1)

        return gt, ms, lms, pan.unsqueeze(0)
    
    def generate_data_from_gt(self, gt):

        pan = torch.mean(gt, dim=-3)
        ms = self.downsampling(self.conv(gt).unsqueeze(0)).squeeze(0)
        lms = self.upsampling(ms.unsqueeze(0)).squeeze(0) # Le falta un self.conv?
        
        return pan, ms, lms


    @staticmethod
    def get_n_channels():
        return 8
    
    @staticmethod
    def get_sampling_factor():
        return 4
    
    @staticmethod
    def show_dataset_image(img):
        # indices (R: 4, G: 2, B: 1)
        return img[..., [4,2,1], :, :]
    

if __name__ == '__main__':
    dataset_train = WorldView3Dataset(os.environ.get('DATASET_PATH'), 'train', 4)
    dataset_validation = WorldView3Dataset(os.environ.get('DATASET_PATH'), 'validation', 4)
    dataset_test = WorldView3Dataset(os.environ.get('DATASET_PATH'), 'test', 4)

    print(f"Test shape: {dataset_test[0][0].shape}")

    print(
        f"Training samples: {len(dataset_train)}\n" \
        f"Validation samples: {len(dataset_validation)}\n" \
        f"Test samples: {len(dataset_test)}\n" 
    )