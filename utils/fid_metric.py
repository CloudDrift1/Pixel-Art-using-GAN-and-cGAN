import torch
from torch import nn
from torchvision.models import inception_v3,Inception_V3_Weights
import cv2
import numpy as np
import os
import scipy 
from utils import *


class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float16 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float16
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations

def calculate_fid(inception_network,images1, images2,  batch_size, device='cuda'):
    """ Calculates the FID between images1 and images2
    Args:
        images1: np.array, shape: (N, H, W, 3), dtype: np.float16 between 0-1 or np.uint8
        images2: np.array, shape: (N, H, W, 3), dtype: np.float16 between 0-1 or np.uint8
        batch size: batch size used for inception network
        device : 'cuda' or 'cpu'
    Returns:
        fid  : float, Frechet Inception Distance between images1 and images2
    """
    assert (images1.shape == images2.shape), "The shapes of two image sets are not matched. {} != {}".format(images1.shape, images2.shape)
    assert (len(images1) >= batch_size), "Batch size {} is larger than the dataset size {}".format(batch_size, len(images1))
    images1 = preprocess_images(images1)
    images2 = preprocess_images(images2)
    mu1, sigma1 = get_statistics(inception_network,images1, batch_size,device=device)
    mu2, sigma2 = get_statistics(inception_network,images2, batch_size,device=device)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

def get_activations(inception_network,images, batch_size, device='cuda'):
    """
    Calculates activations for inception_network for all images
    Args:
        inception_network: Instance of PartialInceptionNetwork
            - It returns the activations (B,2048)
        Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float16
        batch size: batch size used for inception network
            - In general, the batch size of image is larger than the `batch size`. 
            Thus, the input images are split into several batches with size `batch_size` and fed into the network.
        device: 'cuda' or 'cpu'
    Returns:
        activations : numpy array shape: (N, 2048)
    """
    assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                              ", but got {}".format(images.shape)

    num_images = images.shape[0]
    inception_network = inception_network.to(torch.float16).to(device)
    inception_network.eval()
    inception_activations = None

    for i in range(0, num_images, batch_size):
        batch_images = images[i:i+batch_size]
        batch_images = batch_images.to(torch.float16).to(device)

        with torch.no_grad():
            batch_activations = inception_network(batch_images)

        batch_activations = batch_activations.cpu().numpy()
        if inception_activations is None:
            inception_activations = batch_activations
        else:
            inception_activations = np.concatenate((inception_activations, batch_activations), axis=0)

    assert inception_activations.shape == (num_images, 2048), \
        f"Expexted output shape to be: {(num_images, 2048)}, but was: {inception_activations.shape}"
    
    return inception_activations


def get_statistics(inception_network,images, batch_size,device='cuda'):
    """Calculates the statistics used by FID
    Args:
        images: torch.tensor, shape: (N, 3, H, W)
        batch_size: batch size to use to calculate inception scores
    Returns:
        mu:     mean over all activations of the inception model
        sigma:  covariance matrix over all activations of the inception model.

    """

    act = get_activations(inception_network, images, batch_size, device=device)

    #mean
    mu = np.mean(act, axis=0)

    #covariance matrix
    sigma = np.cov(act, rowvar=False)

    assert mu.shape == (2048,), f"Shape mu: {mu.shape}, should be (2048,)"
    if len(act) == 1:
        assert sigma.shape == (), f"Shape sigma: {sigma.shape}, should be () : scalar "
    else :
        assert sigma.shape == (len(act), len(act)), f"Shape sigma: {sigma.shape}, should be ({len(act)}, {len(act)})"

    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Params:
    -- mu1   : The sample mean over activations  (type:np.array, shape: (2048,))
    -- mu2   : The sample mean over activations (type:np.array, shape: (2048,))
    -- sigma1: The covariance matrix over activations 
                (type:np.array, shape: (batch size, batch size))
    -- sigma2: The covariance matrix over activations 
                (type:np.array, shape: (batch size, batch size))

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    epsilon = 1e-6
    sqrt_sigma1 = scipy.linalg.sqrtm(sigma1 + epsilon * np.eye(sigma1.shape[0]))  
    sqrt_sigma2 = scipy.linalg.sqrtm(sigma2 + epsilon * np.eye(sigma2.shape[0]))

    #trace term
    trace_term = np.trace(sigma1 + sigma2 - 2.0 * sqrt_sigma1 @ sqrt_sigma2)

    #FID
    FID = np.linalg.norm(diff) ** 2 + trace_term

    return FID
