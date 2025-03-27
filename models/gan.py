import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
from PIL import Image
from utils.utils import *
from utils.fid_metric import calculate_fid,PartialInceptionNetwork

# import torch.nn.utils.spectral_norm as spectral_norm


class Generator(nn.Module):
    def __init__(self,latent_dim = 100, nc = 3):
        """"
        Args:
            input_shape : shape of the input image
            latent_dim : size of the latent z vector
            nc : number of channels in the training images. For color images this is 3
        """
        self.latent_dim = latent_dim
        super().__init__()

        self.nc = nc
        self.fc = nn.Linear(self.latent_dim, self.nc * 16 * 16)  

        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),

        )

    def forward(self, input):
        """
        Forward pass of the generator.

        Args:
            input : the input to the generator (latent vector) (type : torch.Tensor, size : (batch_size, latent_dim))
        Returns:
            generated_img : the output of the generator (image) (type : torch.Tensor, size : (batch_size, 3, 16, 16))
        """
        assert input.shape == (input.shape[0], self.latent_dim), f"input shape must be (batch_size, latent_dim), not {input.shape}"
        
        generated_img = None

        input = input.view(input.size(0), -1)
        x = self.fc(input)
        x = x.view(x.size(0), 3, 16, 16)  # Reshape to (batch_size, channels, height, width)
        x = self.conv_transpose(x)
        generated_img = x.view(input.size(0), self.nc, 16, 16)

        return generated_img

class Discriminator(nn.Module):
    def __init__(self, nc=3):
        """
        Args:
            256 : size of feature maps in generator
            nc : number of channels in the training images. For color images this is 3
        """
        super().__init__()
        self.nc = nc

        self.model = nn.Sequential(
            nn.Conv2d(self.nc, 64, kernel_size=2, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )


    def forward(self, input):
        """
        Forward pass of the discriminator.
        
        Args:
            input : the input to the discriminator (image) (type : torch.Tensor, size : (batch_size, 1, 16, 16))
        Returns:
            output : the output of the discriminator (probability of being real) (type : torch.Tensor, size : (batch_size))
        """
        assert input.shape == (input.shape[0], self.nc, 16, 16), f"input shape must be (batch_size, 1, 16, 16), not {input.shape}"

        output = self.model(input)
        output = output.view(input.shape[0], -1)

        return output
    