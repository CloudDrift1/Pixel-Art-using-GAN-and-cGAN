# GAN and cGAN for Image Generation with 16x16 Sprites

This repository contains a custom implementation of a Generative Adversarial Network (GAN) and Conditional Generative Adversarial Network (cGAN) for generating images using a 16x16 sprite dataset. The project also includes a custom implementation and testing of the FID metric, a DataLoader, and a Trainer for handling model training and evaluation.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
  - [Generator](#generator)
  - [Discriminator](#discriminator)
  - [cGAN vs GAN](#cgan-vs-gan)
- [Training](#training)
- [Evaluation](#evaluation)
  - [FID Metric](#fid-metric)
- [Usage](#usage)


## Introduction
This project explores image generation through GAN and cGAN techniques. The generator creates images based on random latent vectors, with the cGAN incorporating class labels to condition the generation process. The models are evaluated using the Fréchet Inception Distance (FID) metric.

## Requirements

## Installation
Clone the repository and install the necessary dependencies.

```bash
pip install -r requirements.txt
```
## Dataset
The dataset used in this project consists of 16x16 sprites, which are small images typically used for testing generative models. You can replace this with your own dataset or use a similar one.

The images look as follows,

![image](https://github.com/user-attachments/assets/65bf2e27-5554-4b45-bbfb-c9e9e2d9c372)

## Model Architecture
### Generator
The generator architecture consists of fully connected layers followed by transposed convolution layers to upscale the latent vector into a 16x16 image. It optionally incorporates class labels to condition the image generation process in the case of cGAN.

### Discriminator
The discriminator takes in the generated or real image and attempts to classify whether the image is real or fake. It uses convolution layers with LeakyReLU activations and BatchNorm for stability.

### cGAN vs GAN
GAN: The generator creates images from random noise vectors, while the discriminator tries to distinguish between real and fake images.

cGAN: In addition to the noise vector, the generator receives a class label to generate images that correspond to that class. The discriminator also uses the class label to help distinguish between real and fake images.

## Training
To train the model, you can use the TrainGAN class from train.py, which handles the training loop, loss computation, and model updates. The training process will save the model checkpoints, and you can monitor the progress using the generated Gifs.

## Evaluation
### FID Metric
The Fréchet Inception Distance (FID) is used to evaluate the quality of generated images. The metric computes the distance between the distributions of real and generated images using feature embeddings from a pretrained Inception model. It is implemented in the fid.py file.

To calculate the FID score, run the following command:
```python
fid = calculate_fid(real_images, generated_images)
```

## Usage
After training the model, you can generate images by passing latent vectors (random noise) to the generator. Optionally, you can pass class labels for cGAN to generate class-conditioned images.

```python
generator = Generator(latent_dim=100, nc=3)
generator.load_state_dict(torch.load('generator.pth'))  # Replace with your checkpoint path
generator.eval()
latent_dim = 100
batch_size = 16  # Number of images to generate
z = torch.randn(batch_size, latent_dim)
with torch.no_grad():
    generated_images = generator(z)
```
For cGAN, follow similar procedure but include labels.

```python
labels = torch.randint(0, 4, (batch_size,)) 
one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
with torch.no_grad():
    generated_images = c_generator(z, one_hot_labels)
```


The loss looks as follows,

![image](https://github.com/user-attachments/assets/ea0eebe0-7d84-46db-a3fb-470a15b6883d)


Sample generated images,


![image](https://github.com/user-attachments/assets/813be689-9e8b-4e76-8b43-1100248a610a)


Image from noise process (sample only),


![image](https://github.com/user-attachments/assets/9567aac9-7735-4524-a18e-16f55eef92ba)


