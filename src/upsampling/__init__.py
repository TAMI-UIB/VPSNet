from .upsamplings import *

dict_upsamplings = {
    "bicubic": [Upsampling, Downsampling],
    "pgcu": [Upsampling, Downsampling],
    "zero-padding": [Upsampling0, Downsampling0],
    "conv": [UpsamplingConv, DownsamplingConv],
    "malisat": [UpSamp_4_2, Downsamp_4_2],
    "malisat2": [UpSamp_2_2, Downsamp_2_2],
    "pixel_shuffle": [PSUpsampling, PSDownsampling]
}
