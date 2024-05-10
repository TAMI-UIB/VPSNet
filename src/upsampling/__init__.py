from .bicubic import Upsampling, Downsampling
from .mhfnet import UpSamp_4_2, UpSamp_2_2, Downsamp_4_2, Downsamp_2_2
from .up_conv import UpsamplingConv, DownsamplingConv

dict_upsamplings = {
    "bicubic": [Upsampling, Downsampling],
    "pgcu": [Upsampling, Downsampling],
    "conv": [UpsamplingConv, DownsamplingConv],
    "mhfnet": [UpSamp_4_2, Downsamp_4_2],
    "mhfnet2": [UpSamp_2_2, Downsamp_2_2],
}
