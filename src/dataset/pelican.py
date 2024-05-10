import os
import sys
import dotenv
dotenv.load_dotenv()
sys.path.extend([os.environ.get('PROJECT_PATH')])
from h5py import File
from typing import Literal
import torch
import imageio

from torch.utils.data import Dataset

from upsampling.bicubic import Upsampling, Downsampling, GaussianSmoothing


class PelicanDataset(Dataset):

    IMAGES_TEST_IDX = [6, 11, 12, 14, 20, 22, 24, 26, 27, 37]

    def __init__(self, path_to_dataset: str | os.PathLike, dataset_mode: Literal['train', 'validation', 'test'], sampling_factor = 4, noise_level = None):
        super().__init__()

        self.path = os.path.join(path_to_dataset, "Pelican", "pelican_64.h5")
        self.noise_level = noise_level

        self.dataset = File(self.path)

        self.sampling_factor = sampling_factor

        self.upsampling = Upsampling(4, self.sampling_factor)
        self.downsampling = Downsampling(4, self.sampling_factor)
        self.conv = GaussianSmoothing(4, 9, 1.7)

        self.gt_images = self.dataset[dataset_mode]

    def __getitem__(self, idx):

        gt = torch.from_numpy(self.gt_images[idx]).permute((2,1,0)) / 255

        ms = self.__make_ms(gt)
        lms = self.upsampling(ms.unsqueeze(0)).squeeze(0)
        pan = torch.mean(gt, dim=-3)

        return gt, ms, lms, pan.unsqueeze(0)
    
    @staticmethod
    def get_name():
        return 'pelican'
    
    def __len__(self):
        return len(self.gt_images)

    def __make_ms(self, gt):
        ms = self.downsampling(self.conv(gt).unsqueeze(0)).squeeze(0)

        if self.noise_level is not None:
            ms += torch.normal(0, self.noise_level, size= list(ms.shape))
        
        return ms
    
    @staticmethod
    def get_n_channels():
        return 4
    
    def get_sampling_factor(self):
        return self.sampling_factor

    def __read_image(self, path):
        image_np = imageio.imread(path) / 255
        image_torch = torch.from_numpy(image_np).permute((2,1,0))
        return image_torch
    
    @staticmethod
    def show_dataset_image(img):
        return img[..., [0,1,2], :, :]
    
if __name__ == '__main__':
    dataset_train = PelicanDataset("/home/marctomas/Escritorio/datasets", 'train', 4, None)
    dataset_validation = PelicanDataset("/home/marctomas/Escritorio/datasets", 'validation', 4, None)
    dataset_test = PelicanDataset("/home/marctomas/Escritorio/datasets", 'test', 4, None)

    print(
        f"Training samples: {len(dataset_train)}\n" \
        f"Validation samples: {len(dataset_validation)}\n" \
        f"Test samples: {len(dataset_test)}\n" 
    )


    