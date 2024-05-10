# VPSNet
Variational Pansharpening Shallow Network

Repository under construction due to JSTARS' special edition for IGARSS proceedings

## Abstract
Pansharpening seeks to fuse the spatial details of a high-resolution panchromatic image with the accurate color description encoded in a low-resolution multispectral image to generate a high-resolution multispectral image. Classical variational methods are more interpretable and flexible than pure data-driven learning approaches, but their performance is limited by the use of rigid priors. In this paper, we efficiently combine both techniques by introducing a shallow unfolded network. The proposed energy includes the classical observation model for the multispectral data and a constraint that injects the high frequencies of the panchromatic image into the fused product. The resulting optimization scheme is unrolled into a deep-learning framework that learns the regularizing prior, the operators involved in the observation model, and all hyperparameters. A post-processing module based on a residual channel attention block is introduced to improve the spectral quality. The experiments demonstrate that our method achieves state-of-the-art results on Pelican, WorldView-3, and SEN2VENUS datasets, while having fewer learnable parameters.

## Setup

In order to execute the code properly, a `.env` file is needed with the following variables:

```
NUM_WORKERS = '(Number of workers for data loading)'
DATASET_PATH = "(Absolute path to the folder with the datasets)"
SNAPSHOT_PATH = "(Absolute path to the folder where the registers will be stored)"
PROJECT_PATH = '(Absolute path to the folder of the project)'
EPOCHS = '(Number of epochs to execute by default)'
BATCH_SIZE = '(Batch size by default)'
DEVICE = '(Device in which to execute the model)'
EVALUATION_FREQUENCY = '(The number of epochs between metrics computation and image plotting)'
```

## Training

## Inference

## Citation

