from torch import Tensor
import torch
import math
import numbers
import torch.nn as nn
import torch.nn.functional as F

from src.upsampling.edge_protector import EdgeProtector

###################################################
#           Zero-padding upsamplings              #
###################################################


class Upsampling0(nn.Module):

    def __init__(self, sampling_factor: int, device, fill: float = 0.0) -> None:
        super().__init__()
        self.sampling_factor = sampling_factor
        self.device = device
        self.fill = fill

    def forward(self, u):
        up = (torch.zeros(u.shape[0], u.shape[1], self.sampling_factor*u.shape[2],
              self.sampling_factor*u.shape[3]) + self.fill).to(self.device)
        up[:, :, 0:-1:self.sampling_factor, 0:-1:self.sampling_factor] = u
        return up


class Downsampling0(nn.Module):

    def __init__(self, sampling_factor: int, device, fill: int = 0) -> None:
        super().__init__()
        self.sampling_factor = sampling_factor
        self.device = device

    def forward(self, u):
        if (len(u.shape) == 4):
            down = u[:, :, 0:-1:self.sampling_factor,
                     0:-1:self.sampling_factor]
        else:
            down = u[:, 0:-1:self.sampling_factor, 0:-1:self.sampling_factor]

        return down.to(self.device)

###################################################
#              Bicubic upsamplings                #
###################################################


class Upsampling(nn.Module):

    def __init__(self, n_channels: int, sampling_factor: int) -> None:
        super().__init__()

        self.sampling_factor = sampling_factor

    def forward(self, img):
        return F.interpolate(img, scale_factor=self.sampling_factor, mode='bicubic')

class Downsampling(nn.Module):

    def __init__(self, n_channels: int, sampling_factor: int) -> None:
        super().__init__()

        self.sampling_factor = 1/sampling_factor

    def forward(self, img):
        return F.interpolate(img, scale_factor=self.sampling_factor, mode='bicubic')

###################################################
#          Gaussian kernel Convolution            #
###################################################

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()

        self.padding = kernel_size // 2

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)
        self.groups = channels
        if dim == 2:
            self.conv = nn.Conv2d(in_channels=channels,
                                  out_channels=channels,
                                  kernel_size=kernel_size,
                                  groups=channels,
                                  padding='same',
                                  bias=False,
                                  padding_mode='replicate')

            self.conv.weight.data = kernel
            self.conv.weight.requires_grad = False

        else:
            raise RuntimeError(
                'Only 2 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input)

class UpsamplingConv(nn.Module):

    def __init__(self, n_channels, sampling_factor) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=sampling_factor, stride = sampling_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(x)
    
class DownsamplingConv(nn.Module):

    def __init__(self, n_channels, sampling_factor) -> None:
        super().__init__()
        self.down = nn.Conv2d(n_channels, n_channels, kernel_size=sampling_factor, stride = sampling_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)

class PSUpsampling(nn.Module):

    def __init__(self, n_channels: int, sampling_factor: int) -> None:
        
        self.sampling_factor = sampling_factor

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels*sampling_factor, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=n_channels*sampling_factor, out_channels=n_channels*sampling_factor*sampling_factor, kernel_size=3, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(self.sampling_factor)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        out = self.pixel_shuffle(x_conv2)

        return self.relu(out)

class PSDownsampling(nn.Module):
    
    def __init__(self, n_channels: int, sampling_factor: int) -> None:
        self.sampling_factor = sampling_factor

        self.pixel_unshuffle = nn.PixelUnshuffle(self.sampling_factor)

        self.conv1 = nn.Conv2d(in_channels=n_channels*self.sampling_factor**2, out_channels=n_channels*self.sampling_factor, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=n_channels*self.sampling_factor, out_channels=n_channels, kernel_size=3, padding=1, bias=True)

        self.relu = nn.ReLU()


    def forward(self, x: Tensor) -> Tensor:

        x_shuffled = self.pixel_unshuffle(x)
        x_conv1 = self.conv1(x_shuffled)
        out = self.conv2(x_conv1)

        return self.relu(out)

class UpSamp_4_2(nn.Module):
    def __init__(self, n_channels, sampling_factor, kernel_size=3, depthwise_coef=1, features=3):
        super(UpSamp_4_2, self).__init__()
        self.conv2dTrans = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
        )
        self.conv2dTrans_factor2 = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=5,
            stride=2,
            padding=5 // 2,
            output_padding=1,
        )
        self.edge_protector1 = EdgeProtector(n_channels, 1, kernel_size, features=features)
        self.edge_protector2 = EdgeProtector(n_channels, 1, kernel_size, features=features)
        self.conv2d = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * depthwise_coef,
            kernel_size=3,
            padding=1,
            groups=n_channels,
        )
        self.conv2d.weight.data = (1 / 16) * torch.ones(self.conv2d.weight.data.size())

    def forward(self, input, support_d2, support_full):

        input = self.conv2dTrans(input)

        input = self.edge_protector1(input, support_d2 / 10)

        input = self.conv2dTrans_factor2(input)
        input = self.edge_protector2(input, support_full / 10)

        input = self.conv2d(input)

        return input

class Downsamp_4_2(nn.Module):
    def __init__(self, channels, sampling_factor,  depthwise_multiplier=1):
        super(Downsamp_4_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels * depthwise_multiplier, kernel_size=5,
                               padding=5 // 2, groups=channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels * depthwise_multiplier, kernel_size=5,
                               padding=5 // 2, groups=channels)

    def forward(self, input):
        height = input.size(2)
        width = input.size(3)
        # Downsampling by a factor of 2
        x_d2 = self.conv2(input)[:, :, 0:height: 2, 0:width: 2]
        # Downsampling by a ratio of 2
        x_low = self.conv1(x_d2)[:, :, 0: int(height / 2): 2, 1: int(width / 2): 2]

        return x_low
    
class UpSamp_2_2(nn.Module):
    def __init__(self, n_channels, sampling_factor, kernel_size=3, depthwise_coef=1, features=3):
        super(UpSamp_2_2, self).__init__()
        self.conv2dTrans = nn.ConvTranspose2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
        )
        self.edge_protector = EdgeProtector(n_channels, 1, kernel_size, features=features)
        self.conv2d = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * depthwise_coef,
            kernel_size=3,
            padding=1,
            groups=n_channels,
        )
        self.conv2d.weight.data = (1 / 16) * torch.ones(self.conv2d.weight.data.size())

    def forward(self, input, support_d2, support_full):

        input = self.conv2dTrans(input)

        input = self.edge_protector(input, support_full / 10)

        input = self.conv2d(input)

        return input

class Downsamp_2_2(nn.Module):
    def __init__(self, channels, sampling_factor,  depthwise_multiplier=1):
        super(Downsamp_2_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels * depthwise_multiplier, kernel_size=5,
                               padding=5 // 2, groups=channels)
        

    def forward(self, input):
        height = input.size(2)
        width = input.size(3)
        
        # Downsampling by a factor of 2
        x_low = self.conv1(input)[:, :, 0:height: 2, 0:width: 2]

        return x_low
