import os
from typing import Literal
import torch
import numpy as np
import csv
from torch.utils.data import Dataset
import sys
sys.path.extend(['/home/marctomas/Escritorio/Repositoris/E2EPansharpening'])
from src.utils.upsamplings import Upsampling, Downsampling, GaussianSmoothing

class SEN2VENUSDataset(Dataset):
    def __init__(self, data_path: str | os.PathLike, fold: Literal['train', 'validation', 'test'] , n_channels: Literal[4, 8] = 4, train_prop: float = 0.8, val_prop: float = 0.1, test_prop: float = 0.1, kernel_size: int = 9, std: float= 1.7, seed: int = 2024) -> None:
        self.n_channels = n_channels
        self.data_path = data_path +"/SEN2VENUS"
        self.seed = seed
        self.fold = fold
        self.train_prop, self.val_prop, self.test_prop = train_prop, val_prop, test_prop
        self.sites = ['ATTO', 'ES-LTERA', 'ESTUAMAR', 'FGMANAUS', 'FR-LQ1', 'K34-AMAZ', 'MAD-AMBO', 'SO2', 'SUDOUE-4']

        self.dataset = None

        self.gaussian_convolution = GaussianSmoothing(self.n_channels, kernel_size, std)

        self.upsampling = Upsampling(self.n_channels, sampling_factor=2)
        self.downsampling = Downsampling(self.n_channels, sampling_factor=2)
       
        self.patch_list = self.__get_split()

    def __len__(self):
       return len(self.patch_list)

    def __getitem__(self, idx):
        
        site, tile_id, num_patch = self.patch_list[idx]

        images_path = os.path.join(self.data_path, site)

        venus_2348_file = os.path.join(images_path, f'{tile_id}_05m_b2b3b4b8.pt')
        sen2_2348_file = os.path.join(images_path, f'{tile_id}_10m_b2b3b4b8.pt')
        pan_file = os.path.join(images_path, f'{tile_id}_PAN.pt')

        venus_2348 = torch.load(venus_2348_file)[num_patch, :, :, :]
        venus_2348 = venus_2348 / 10000

        sen2_2348 = torch.load(sen2_2348_file)[num_patch, :, :, :]
        sen2_2348 = sen2_2348 / 10000

        pan = torch.load(pan_file)[num_patch, :, :, :]
        P = pan / 10000

        M = sen2_2348
        U = venus_2348

        if self.n_channels == 8:
            venus_5678a_file = os.path.join(images_path, f'{tile_id}_05m_b4b5b6b8a.pt')
            sen2_5678a_file = os.path.join(images_path, f'{tile_id}_20m_b4b5b6b8a.pt')

            venus_5678a = torch.load(venus_5678a_file)[num_patch, :, :, :]
            venus_5678a = venus_5678a / 10000

            sen2_5678a = torch.load(sen2_5678a_file)[num_patch, :, :, :]
            sen2_5678a = sen2_5678a / 10000

            M = self.__make_M(sen2_2348, sen2_5678a)
            U = self.__make_U(venus_2348, venus_5678a)

        

        # return gt, ms, lms, pan
        return U, M, self.upsampling(M.unsqueeze(0)).squeeze(0), P 

    def __gen_patches_full_list(self):
        patches_list = []
        for site in self.sites:
            images_path = os.path.join(self.data_path, site)
            with open(f'{images_path}/index.csv') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    num_patches = int(row['nb_patches'])
                    site_name = row['vns_site']
                    mgrs_tile = row['s2_tile']
                    adquisition_date = row['date']
                    tile_id = f'{site_name}_{mgrs_tile}_{adquisition_date}'
                    for idx in range(num_patches):
                        patches_list.append((site, tile_id, idx))
        return patches_list

    def __get_split(self):
        patches_list = self.__gen_patches_list()
        np.random.seed(self.seed)
        np.random.shuffle(patches_list)
        train_split_point = int(self.train_prop * len(patches_list))
        val_split_point = train_split_point + int(self.val_prop * len(patches_list))
        test_split_point = val_split_point + int(self.test_prop * len(patches_list))

        train_patches_list = patches_list[:train_split_point]
        val_patches_list = patches_list[train_split_point:val_split_point]
        test_patches_list = patches_list[val_split_point:test_split_point]

        if self.fold == 'train':
            return train_patches_list
        elif self.fold == 'validation':
            return val_patches_list
        elif self.fold == 'test':
            return test_patches_list
        


    def __make_M(self, sen2_2348, sen2_5678a):
        M = torch.cat((sen2_2348, self.upsampling(sen2_5678a.unsqueeze(0)).squeeze(0)), dim=0)
        return M
    
    def DB(self, u: torch.Tensor) -> torch.Tensor:
        return self.downsampling(self.gaussian_convolution(u))
        
    def __make_U(self, venus_2348, venus_5678a):
        U = torch.cat((venus_2348, venus_5678a), dim=0)
        return U
    
    def get_n_channels(self):
        return self.n_channels
    
    @staticmethod
    def get_sampling_factor():
        return 2
    
    @staticmethod
    def show_dataset_image(img):
        return img[..., [2, 1, 0], :, :]

if __name__ == '__main__':
    dataset = SEN2VENUSDataset("/home/marctomas/Escritorio/datasets", 'train')
    
    gt_nans = []
    ms_nans = []
    lms_nans = []
    pan_nans = []

    for i in range(len(dataset)):
        gt, ms, lms, pan = dataset[i]

        gt_nans.append(torch.any(torch.isnan(gt)))
        ms_nans.append(torch.any(torch.isnan(ms)))
        lms_nans.append(torch.any(torch.isnan(lms)))
        pan_nans.append(torch.any(torch.isnan(pan)))
            
    print(f'GT: {np.any(gt_nans)}')
    print(f'MS: {np.any(ms_nans)}')
    print(f'LMS: {np.any(lms_nans)}')
    print(f'PAN: {np.any(pan_nans)}')