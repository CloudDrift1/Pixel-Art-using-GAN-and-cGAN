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


class cGenerator(nn.Module):
    def __init__(self,input_shape = (1,28,28),
          latent_dim = 100,  num_classes = 10, nc = 3):
        """"
        Args:
            input_shape : shape of the input image
            latent_dim : size of the latent z vector
            num_classes : number of classes
            nc : number of channels in the training images. For color images this is 3
        """
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        super(cGenerator, self).__init__()

        self.nc = nc
        self.fc = nn.Linear(self.latent_dim + self.num_classes, self.nc * 16 * 16)  
    
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
        

    def forward(self, input,label):
        """
        Args:
            input : random noise z
            label : label for the image
        Returns:
            generated image
        """
        batch_size = input.shape[0]  

        assert input.shape[0] == label.shape[0], "Batch size of input and label must match"
        assert label.shape[1] == self.num_classes, f"Label shape must be (batch_size, {self.num_classes}), not {label.shape}"

        #concatenate noise and label
        input = torch.cat((input, label), dim=1)

        gen_img = self.fc(input)
        gen_img = gen_img.view(gen_img.size(0), 3, 16, 16)  # Reshape to (batch_size, channels, height, width)
        gen_img = self.conv_transpose(gen_img)
        output = gen_img.view(input.size(0), self.nc, 16, 16)

        return output
  
    
class cDiscriminator(nn.Module):
    def __init__(self,input_shape = (3,16,16), nc = 3, num_classes=10):
        """"
            input_shape : shape of the input image
            nc : number of channels in the training images. For color images this is 3
            num_classes : number of classes
        """
        super(cDiscriminator, self).__init__()

        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.num_classes = num_classes
        self.nc = nc
        self.label_embedding = nn.Embedding(self.num_classes, self.input_size//self.nc)

        self.model = nn.Sequential(
            nn.Conv2d(self.nc + self.num_classes, 64, kernel_size=2, stride=2, padding=1),
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
        

    def forward(self, input,label): 
        assert input.shape == (input.shape[0], self.nc, 16, 16), f"input shape must be (batch_size, 1, 16, 16), not {input.shape}"

        embedded_label = self.label_embedding(label.to(torch.long)).view(label.shape[0], self.num_classes, self.input_shape[-1], self.input_shape[-1])

        #concatenate image and embedded label
        input = torch.cat([input, embedded_label], dim=1)
        output = self.model(input)
        output = output.view(input.shape[0], -1)
        
        return output