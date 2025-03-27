import os
from turtle import width
import torch
import torch.nn as nn
import random
import numpy as np
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
from types import SimpleNamespace
import platform,socket,re,uuid,json,psutil,logging
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



def set_randomness(seed : int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_one_iter_train_GAN(training_GAN,config,device):
    images = torch.load('./utils/test_file/test_images.pth',map_location=device)
    labels = torch.load('./utils/test_file/test_labels.pth',map_location=device)
    noise = torch.load("./utils/test_file/test_noise.pth",map_location=device)[:config.batch_size,:]
    
    test_data = torch.load('./utils/test_file/test_GAN.pth',map_location=device)
    from test_GAN_one_iter import Generator, Discriminator
    generator = Generator(latent_dim = config.latent_dim,nc = config.nc).to(device)
    discriminator = Discriminator(nc=config.nc).to(device)
    generator.load_state_dict(test_data['generator_weight'])
    discriminator.load_state_dict(test_data['discriminator_weight'])
    trainer = training_GAN(train_loader = None, \
                                generator = generator, discriminator = discriminator, \
                                device=device, config = config)

    results = trainer.one_iter_train(images,labels,noise)
    return results


def visualizeGAN():
    images = []
    epochs = [0,1,5,10]
    for i in epochs:
        img = cv2.imread(f'./utils/test_file/GAN_00{i:02d}_sprite.png')
        images.append(img)
    for i in range(len(epochs)):
        plt.subplot(1,len(epochs),i+1)
        plt.title(f"EPOCH 0{epochs[i]:02d}")
        plt.axis('off')
        plt.imshow(images[i])


def show_image_with_GAN(gen,config=None,device='cuda',cols=4,rows=4):
    if config is None:
        config = SimpleNamespace(
                latent_dim = 100
        )
    with torch.no_grad():
        fixed_noise = torch.randn(cols*rows, config.latent_dim).to(device)
        img_fake = gen(fixed_noise).detach().cpu()
        fig = plt.figure(figsize=(6,6))
        for i in range(rows * cols):
            idx_img_fake = img_fake[i]
            fig.add_subplot(rows, cols, i+1)
            plt.axis('off')
            
            if len(idx_img_fake.size()) == 3:
                idx_img_fake = idx_img_fake.permute(1,2,0)
            idx_img_fake = (idx_img_fake - idx_img_fake.min())/ (idx_img_fake.max() - idx_img_fake.min())
            plt.imshow(idx_img_fake, cmap='gray')
    plt.show() 

def preprocess_image(im):
    original_im = im.clone()
    transform = transforms.Compose([
        transforms.Resize(299),
    ])
    if im.dtype == torch.uint8:
        im = im.astype(torch.float16)  / 255
    elif im.max() > 1.0 or im.min() < 0.0:
        im = (im - im.min()) / (im.max() - im.min())
    im = im.type(torch.float16)
    im = transform(im)
    if im.shape[0] == 1:
        im = im.repeat(3,1,1)
    # print(im.shape)
    try:
        assert im.max() <= 1.0, im.max()
        assert im.min() >= 0.0, im.min()
        assert im.dtype == torch.float16
        assert im.shape == (3, 299, 299), im.shape
    except:
        print("original_im", original_im.shape)
        print(original_im.max(), original_im.min())
        print("transformed image",im.shape)
        print(im.max(), im.min())
        raise AssertionError 
    return im

def show_image_with_label(gen,config=None,device='cuda',cols=4,rows=4):
    if config is None:
        config = SimpleNamespace(
                latent_dim = 100
        )
    with torch.no_grad():
        fixed_noise = torch.randn(cols*rows, config.latent_dim).to(device)
        label = torch.zeros((cols*rows,5)).to(device)
        for i in range(cols*rows):
            label[i][i%5] = 1
        img_fake = gen(fixed_noise,label).detach().cpu()
        size_of_figure = (int(cols*1.5),int(rows*1.5))
        fig = plt.figure(figsize=size_of_figure)
        for i in range(rows * cols):
            
            fig.add_subplot(rows, cols, i+1)
            plt.title(torch.argmax(label[i]).item())
            plt.axis('off')
            img_fake[i] = (img_fake[i] - img_fake[i].min())/ (img_fake[i].max() - img_fake[i].min())
            plt.imshow(img_fake[i].permute(1,2,0), cmap='gray')
    plt.show()

def get_MNIST_image_and_visualize():
    image1 = torch.Tensor(cv2.imread('./test_file/GAN_0000.png',cv2.IMREAD_GRAYSCALE)).unsqueeze(0).unsqueeze(0)
    image2 = torch.Tensor(cv2.imread('./test_file/GAN_0050.png',cv2.IMREAD_GRAYSCALE)).unsqueeze(0).unsqueeze(0)
    image3 = torch.Tensor(cv2.imread('./test_file/GAN_0100.png',cv2.IMREAD_GRAYSCALE)).unsqueeze(0).unsqueeze(0)
    plt.subplot(131)
    plt.title("EPOCH 000")
    plt.imshow(image1.squeeze())
    plt.subplot(132)
    plt.title("EPOCH 050")
    plt.imshow(image2.squeeze())
    plt.subplot(133)
    plt.title("EPOCH 100")
    plt.imshow(image3.squeeze())
    plt.pause(0.01)
    return image1, image2, image3

def preprocess_images(images):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        images: torch.Tensor, shape: (N, 3, H, W), dtype: float16 between 0-1 or np.uint8

    Return:
        final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float16 between 0-1
    """
    final_images = torch.stack([preprocess_image(im) for im in images], dim=0)
    assert final_images.shape == (images.shape[0], 3, 299, 299)
    assert final_images.max() <= 1.0
    assert final_images.min() >= 0.0
    assert final_images.dtype == torch.float16
    return final_images