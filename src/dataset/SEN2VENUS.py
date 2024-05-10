import os
from typing import Literal
import torch
from torch.utils.data import Dataset
import sys
import os
import sys
import dotenv
dotenv.load_dotenv()
sys.path.extend([os.environ.get('PROJECT_PATH')])
from upsampling.bicubic import Upsampling, Downsampling, GaussianSmoothing

class SEN2VENUSDataset(Dataset):

    IMAGES_TEST_IDX = [35, 65, 67, 77, 78, 79, 91, 93, 111, 117, 122, 126, 175, 177, 178, \
                       180, 181, 209, 219, 240, 244, 317, 444, 447, 496, 566, 570, 640, 656, \
                       688, 722, 725, 729, 784, 790, 881, 935, 965, 966, 967, 971, 999]


    def __init__(self, data_path: str | os.PathLike, fold: Literal['train', 'validation', 'test'], crop_size: Literal[64, 128] =64, kernel_size: int = 9, std: float= 1.7, **kwargs) -> None:
        self.n_channels = 4
        self.fold = fold
        self.data_path = os.path.join(data_path, "SEN2VENUS", f"sen2venus_{self.fold}_{crop_size}.pth")

        self.gaussian_convolution = GaussianSmoothing(self.n_channels, kernel_size, std)
        self.upsampling = Upsampling(self.n_channels, sampling_factor=2)
        self.downsampling = Downsampling(self.n_channels, sampling_factor=2)
       
        self.dataset = torch.load(self.data_path)

    def __len__(self):
        return self.dataset['gt'].shape[0]

    def __getitem__(self, idx):

        gt = self.dataset["gt"][idx].to(torch.float32) / 10000     
        ms = self.dataset["ms"][idx].to(torch.float32) / 10000
        pan = self.dataset["pan"][idx].to(torch.float32) / 10000

        lms = self.upsampling(ms.unsqueeze(0)).squeeze(0)
        
        return gt, ms, lms, pan 

    def DB(self, u: torch.Tensor) -> torch.Tensor:
        return self.downsampling(self.gaussian_convolution(u))

    def get_n_channels(self):
        return self.n_channels
    
    @staticmethod
    def get_name():
        return 'sen2venus'
    
    @staticmethod
    def get_sampling_factor():
        return 2
    
    @staticmethod
    def show_dataset_image(img):
        return img[..., [2, 1, 0], :, :]

if __name__ == '__main__':
    dataset_train = SEN2VENUSDataset("/lhome/ext/uib107/uib107c/datasets/", 'train')
    dataset_validation = SEN2VENUSDataset("/lhome/ext/uib107/uib107c/datasets/", 'validation')
    dataset_test = SEN2VENUSDataset("/lhome/ext/uib107/uib107c/datasets/", 'test')
    
    print(
        f"Training samples: {len(dataset_train)}\n" \
        f"Validation samples: {len(dataset_validation)}\n" \
        f"Test samples: {len(dataset_test)}\n" 
    )