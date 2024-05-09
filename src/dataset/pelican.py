import os
import math
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision

class PelicanDataset(Dataset):
    def __init__(self, path_to_images, crop_size, sampling_factor, kernel_size, std, noise_level = None, flag_training = False, flag_augmentation = False):
        super().__init__()

        self.path = path_to_images
        self.training = flag_training
        self.augmentation = flag_augmentation
        self.noise_level = noise_level

        self.stmichel1 = self.__read_image(self.path + "stmichel2.tif")
        self.stmichel2 = self.__read_image(self.path + "stmichel.tif")

        self.image_size = self.stmichel1.shape[-1]
        self.n_channels = self.stmichel1.shape[-3]
        self.crop_size = crop_size

        self.images_training = math.floor(2*(self.image_size/self.crop_size)**2*0.8)

        self.crops_per_image = self.image_size // self.crop_size  #TODO: Refactor this variable

        self.gaussian_kernel = self.__generate_gaussian_kernel(kernel_size, std, self.n_channels)

        self.gaussian_convolution = torchvision.transforms.Lambda(lambda x: F.conv2d(x, self.gaussian_kernel, padding='same', groups = self.n_channels))

        self.upsampling = torchvision.transforms.Resize(size=crop_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.downsampling = torchvision.transforms.Resize(size=crop_size//sampling_factor, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        # Composed transformations
        self.DB = torchvision.transforms.Compose([self.gaussian_convolution, self.downsampling])
        self.DBtranspose = torchvision.transforms.Compose([self.upsampling, self.gaussian_convolution])
        
        

    def __getitem__(self, idx):
        if self.training:

            if idx < self.crops_per_image**2:
                reference = self.stmichel1 / 255

            else:
                idx = idx - self.crops_per_image**2
                reference = self.stmichel2 / 255
        
        else:
            idx = idx + (self.images_training - self.crops_per_image**2) # imagenes en training - imagenes en stmichel1
            reference = self.stmichel2

        column = idx % self.crops_per_image
        row = idx // self.crops_per_image

        gt = reference[:, row*self.crop_size:(row+1)*self.crop_size, column*self.crop_size:(column+1)*self.crop_size]

        ms = self.__make_ms(gt)
        u_tilde = self.__make_u_tilde(gt)
        pan = self.__make_pan(gt)
        p_tilde = self.__make_p_tilde(pan)

        return gt, ms, u_tilde, pan
        
    def __len__(self):
        return math.floor(2*(self.image_size/self.crop_size)**2*0.8) if self.training else math.ceil(2*(self.image_size/self.crop_size)**2*0.2)
    
    def __make_pan(self, gt):
        pan = torch.sum(gt, dim = 0) / gt.shape[0]
        pan = pan.repeat(self.n_channels, 1, 1)
        return pan
    
    def __make_ms(self, gt):
        ms = self.DB(gt)

        if self.noise_level is not None:
            ms += torch.normal(0, self.noise_level, size= list(ms.shape))
        
        return ms
    
    def get_n_channels(self):
        return 4

    def __make_p_tilde(self, pan):
        p_tilde = self.DBtranspose(self.DB(pan))
        return p_tilde
    
    def __make_u_tilde(self, gt):
        u_tilde = self.DBtranspose(self.DB(gt))

        return u_tilde

    def __read_image(self, path):
        image_np = imageio.imread(path) / 255
        image_torch = torch.from_numpy(image_np).permute((2,1,0))
        return image_torch
    
    @staticmethod
    def show_dataset_image(img):
        return img[..., [0,1,2], :, :]
    
    def __generate_gaussian_kernel(self, kernel_size, sigma, channels):

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        return gaussian_kernel


if __name__ == '__main__':
    std = 1.7
    kernel_size = 4 * math.ceil(std) +1

    dataset_train = PelicanDataset("/lhome/ext/uib107/uib107c/datasets/Pelican/", 32, 4, kernel_size, std, None, True, False)
    dataset_val = PelicanDataset("/lhome/ext/uib107/uib107c/datasets/Pelican/", 32, 4, kernel_size, std, None, False, False)

    gt, pan, ms, p_tilde, u_tilde = dataset_train[5]

    imageio.imwrite("images/gt.tif", gt.permute((2,1,0)).detach().numpy()*255)
    imageio.imwrite("images/pan.tif", pan.permute((2,1,0)).detach().numpy()*255)
    imageio.imwrite("images/ms.tif", ms.permute((2,1,0)).detach().numpy()*255)
    imageio.imwrite("images/p_tilde.tif", p_tilde.permute((2,1,0)).detach().numpy()*255)
    imageio.imwrite("images/u_tilde.tif", u_tilde.permute((2,1,0)).detach().numpy()*255)

    test_image = dataset_train.gaussian_convolution(gt)
    imageio.imwrite("images/test.tif", test_image.permute((2,1,0)).detach().numpy()*255)

    # print("Crops:", dataset_train.crops_per_image**2)

    # print("Length training:", len(dataset_train))
    # print("Length validation:", len(dataset_val))

    # print("Asserting training dataset")

    # for i in range(len(dataset_train)):
    #     gt, pan, ms, p_tilde, u_tilde = dataset_train[i]
    #     print("Index:", i)

    # print()
    # print("Asserting validation dataset")

    # for i in range(len(dataset_val)):
    #     gt, pan, ms, p_tilde, u_tilde = dataset_val[i]
    #     print("Index:", i)
        

    