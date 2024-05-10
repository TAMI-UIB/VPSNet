from torch import Tensor
import torch
import math
import numbers
import torch.nn as nn
import torch.nn.functional as F

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
