import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
import tqdm
import matplotlib.pyplot as plt

import imageio.v2 as imageio
import platform,socket,re,uuid,json,psutil,logging
import random
from turtle import width
from types import SimpleNamespace
from utils.fid_metric import calculate_fid, PartialInceptionNetwork


def getSystemInfo():
    try:
        info={}
        info['platform']=platform.system()
        info['platform-release']=platform.release()
        info['platform-version']=platform.version()
        info['architecture']=platform.machine()
        info['hostname']=socket.gethostname()
        info['ip-address']=socket.gethostbyname(socket.gethostname())
        info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
        info['processor']=platform.processor()
        info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
        return json.dumps(info)
    except Exception as e:
        logging.exception(e)

def loss_function(prob, label='fake'):
    """
    Return the loss function used for training the GAN.
    Args:
        prob : the probability of the input being real or fake (output of the discriminator) 
                (type : torch.Tensor, size : (batch_size, 1))
        label : either 'fake' or 'real' to indicate whether the input is (intend to be) fake or real 
                (type : str)"
    Returns:
        loss : the loss value (type : torch.Tensor, size : torch.Size([]))
    """

    batch_size = prob.shape[0]
    target = torch.zeros_like(prob) if label == 'fake' else torch.ones_like(prob)

    #binary cross-entropy loss
    loss = torch.nn.functional.binary_cross_entropy(prob, target)

    assert loss.shape == torch.Size([]), f"loss shape must be torch.Size([]), not {loss.shape}"
    
    return loss


class train_GAN:
    def __init__(self, train_loader, generator, discriminator, device,
                 config, fid_score_on = False, save_model = False, img_show = False,
                 evaluation_on = False):
        """"
        Initialize the training_GAN class.

        Args:
            train_loader : the dataloader for training dataset (type : torch.utils.data.DataLoader)
            generator : the generator model (type : nn.Module)
            discriminator : the discriminator model (type : nn.Module)
            device : the device where the model will be trained (type : torch.device)
            config : the configuration for training (type : SimpleNamespace)
            fid_score_on : whether to calculate the FID score or not (type : bool)
            save_model : whether to save the model or not (type : bool)
            img_show : whether to show the generated image or not for each epoch (type : bool)
            evaluation_on : whether to evaluate the model or not (type : bool). It turns on for the last epoch.
        """
        self.train_loader = train_loader
        self.num_epochs = config.epoch
        self.lr = config.lr
        self.latent_dim = config.latent_dim
        self.batch_size = config.batch_size
        self.device = device
        self.fid_score_on = fid_score_on
        self.img_show = img_show
        self.evaluation_on = evaluation_on
        self.config = config

        self.generated_img = []
        self.G_loss_history = []
        self.D_loss_history = []
        self.system_info = getSystemInfo()
        self.save_model = save_model
        self.model_name = 'GAN'
        self.generator = generator
        self.discriminator = discriminator
    
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def make_gif(self):
        """
        Save the generated images as a gif file.
        """
        if len(self.generated_img) <= 1:
            print("No frame to save")
            return
        else :
            print("Saving gif file...")
            for i in range(len(self.generated_img)):
                self.generated_img[i] = Image.fromarray(self.generated_img[i])
            self.generated_img[0].save(f"./{self.model_name}_generated_img.gif",
                                save_all=True, append_images=self.generated_img[1:], 
                                optimize=False, duration=700, loop=1) 
    def one_iter_train(self, images,label,noise=None):
        """
        Train the GAN model for one iteration.

        Args:
            images : the real images (type : torch.Tensor, size : (batch_size, 1, 16, 16))
            label : the label of the real images (type : torch.Tensor, size : (batch_size))
            noise : the random noise z (type : torch.Tensor, (batch_size, latent_dim)
                - If noise is None, then generate the random noise z inside the function. 
                - In general, noise is None. It is for testing the model with fixed noise,
        Returns:
            loss_G : the loss value for generator (type : float)
            loss_D : the loss value for discriminator (type : float)
        """

        if noise is None:
            noise = torch.randn(images.size(0), self.latent_dim, device=self.device)

        #discriminator
        self.discriminator.zero_grad()
        fake_images = self.generator(noise)
        prob_real = self.discriminator(images)
        prob_fake = self.discriminator(fake_images)

        loss_D_real = loss_function(prob_real, label='real')
        loss_D_fake = loss_function(prob_fake, label='fake')
        loss_D = 0.5*(loss_D_real + loss_D_fake) 

        loss_D.backward()
        self.optimizer_D.step()

        #generator
        self.generator.zero_grad()
        fake_images = self.generator(noise)
        prob_fake_updated = self.discriminator(fake_images)
        loss_G = loss_function(prob_fake_updated, label='real')
        loss_G.backward()
        self.optimizer_G.step()

        return {
                "loss_G" : loss_G.item(),
                "loss_D" : loss_D.item()
                }
    
    def get_fake_images(self, z,labels):
        with torch.no_grad():
            out = self.generator(z)
        return out
    
    def FID_score(self,network, test_images, fake_images, batch_size):
        from fid_score import FID_score
        fid = FID_score(network,test_images,fake_images, batch_size)
        return fid
    
    def train(self):
        """
        Train the GAN model.
        """
        if self.fid_score_on:
            inception_network = PartialInceptionNetwork().cuda()
        else:
            inception_network = None
        try : 
            test_noise = torch.load("./utils/test_file/test_noise.pth", map_location=self.device)
            test_batch_size = test_noise.shape[0]
            test_data = torch.load("./utils/test_file/img_per_label_sprite.pth", map_location=self.device)
            
            for epoch in range(1,self.num_epochs+1):
                pbar = tqdm.tqdm(enumerate(self.train_loader,start=1), total=len(self.train_loader))
                for i, (images, labels) in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    if epoch == 1 and i == 1:
                        # save the generated images before training
                        fake_images = self.get_fake_images(test_noise,test_data[5]['label'])
                        grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True).detach().cpu().permute(1,2,0).numpy()
                        self.generated_img.append((grid_img* 255).astype('uint8'))
                    self.generator.train()
                    self.discriminator.train()
                    results = self.one_iter_train(images,labels)
                    self.generator.eval()
                    self.discriminator.eval()
                    loss_G, loss_D = results['loss_G'], results['loss_D']
                    self.G_loss_history.append(loss_G)
                    self.D_loss_history.append(loss_D)
                    

                    pbar.set_description(
                        f"Epoch [{epoch}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Loss_D: {loss_D:.6f}, Loss_G: {loss_G:.6f}")
                
                
                fake_images = self.get_fake_images(test_noise,test_data[5]['label'])
                grid_img = torchvision.utils.make_grid(fake_images[:16], nrow=4, normalize=True).detach().cpu().permute(1,2,0).numpy()
                self.generated_img.append((grid_img* 255).astype('uint8'))
                # calculate FID score and show the generated images
                if self.model_name == 'GAN':
                    if self.img_show is True:
                        plt.axis('off')
                        plt.imshow(grid_img)
                        plt.pause(0.01)
                    if self.fid_score_on :
                        # custom fid score
                        fid = calculate_fid(inception_network, test_data[5]['img'],fake_images, 16)
                elif self.model_name == 'cGAN':
                    if self.fid_score_on :
                        fid = 0.0
                    for label_idx in range(self.num_classes):
                        test_label = torch.zeros(test_batch_size,self.num_classes, device=self.device) 
                        test_label[:,label_idx] = 1
                        fake_images = self.get_fake_images(test_noise,test_label)
                        if self.img_show is True:
                            plt.subplot(1,self.num_classes,label_idx+1)
                            plt.axis('off')
                            plt.title(f"label : {label_idx}")
                            plt.imshow(torchvision.utils.make_grid(fake_images[label_idx][:16], nrow=4, normalize=True).detach().cpu().permute(1,2,0).numpy())
                        if self.fid_score_on :
                            fid += calculate_fid(inception_network, test_data['im'][test_label],fake_images, 16) / self.num_classes
                    
                    if self.img_show is True:
                        plt.pause(0.01)
                else:
                    raise ValueError("model_name must be either GAN or cGAN, not",self.model_name)
                
                if self.fid_score_on :
                    pbar.write(f"EPOCH {epoch} - FID score : {fid:.6f}")
                
        except KeyboardInterrupt:
            print('Keyboard Interrupted, finishing training...')
        
        # TODO : Save the model
        
        if self.evaluation_on is True:
            if inception_network is None:
                inception_network = PartialInceptionNetwork().cuda()
            if self.model_name == 'GAN':
                fake_images = self.get_fake_images(test_noise,test_data[5]['label'])
                self.fid = self.FID_score(inception_network,test_data[5]['img'],fake_images, 16)
            elif self.model_name == 'cGAN':
                fid = 0.0
                for label_idx in range(self.num_classes):
                    test_label = torch.zeros(test_batch_size,self.num_classes, device=self.device) 
                    test_label[:,label_idx] = 1
                    fake_images = self.get_fake_images(test_noise,test_label)
                    fid += self.FID_score(inception_network,test_data[label_idx],fake_images, 16) / self.num_classes
                self.fid = fid
            else : 
                raise ValueError("model_name must be either GAN or cGAN, not",self.model_name)
        
        if self.evaluation_on is True:
            pbar.write(f"EPOCH {epoch} - FID score : {self.fid:.6f} (evaluation)")
        
        if self.save_model is True:
            self.save_results()
        return {'generator' : self.generator,
                'generator_state_dict' : self.generator.state_dict(),
                'discriminator' : self.discriminator,
                'discriminator_state_dict' : self.discriminator.state_dict(),
                'G_loss_history' : self.G_loss_history,
                'D_loss_history' : self.D_loss_history}
    def save_results(self):
        """
        Save the trained model.
        """
        data = {
            "generator" : self.generator.state_dict(),
            "discriminator" : self.discriminator.state_dict(),
            "system_info" : self.system_info,
            "epoch" : self.num_epochs,
            "generated_img" : self.generated_img,
        }
        if self.evaluation_on is True:
            data["fid_score"] = self.fid
        if self.save_model is True:
            self.make_gif()
            torch.save(data, f"./{self.model_name}_model.pth")

class train_cGAN(train_GAN):
    def __init__(self, train_loader, generator, discriminator, device, config, fid_score_on=False,save_model=False,img_show=False,evaluation_on=False):
        super().__init__(train_loader, generator, discriminator, device, config, fid_score_on,save_model,img_show,evaluation_on)
        self.num_classes = config.num_classes
        self.model_name = 'cGAN'
    def get_fake_images(self, z, labels):
        return self.generator(z,labels)
    def one_iter_train(self,images,label,noise=None):
        """
        Args:
            images : the real images (type : torch.Tensor, size : (batch_size, 1, 16, 16))
            label : the label of the real images (type : torch.Tensor, size : (batch_size))
            noise : the random noise z (type : torch.Tensor, size : (batch_size, latent_dim))
                - If noise is None, then generate the random noise z inside the function.
                - In general, noise is None. It is for testing the model with fixed noise,
        """

        #random noise if not provided
        if noise is None:
            noise = torch.randn(images.size(0), self.latent_dim, device=self.device)

        self.discriminator.zero_grad()
        fake_images = self.generator(noise, label)
        prob_real = self.discriminator(images, label)
        prob_fake = self.discriminator(fake_images.detach(), label)

        #discriminator loss
        loss_D_real = loss_function(prob_real, label='real')
        loss_D_fake = loss_function(prob_fake, label='fake')
        loss_D = 0.5*(loss_D_real + loss_D_fake)

        #update discriminator
        loss_D.backward()
        self.optimizer_D.step()

        #update generator
        self.generator.zero_grad()
        fake_images = self.generator(noise, label)

        prob_fake_updated = self.discriminator(fake_images, label)

        #generator loss
        loss_G = loss_function(prob_fake_updated, label='real')

        #backprop and update generator
        loss_G.backward()
        self.optimizer_G.step()

        return {
                "loss_G" : loss_G.item(),
                "loss_D" : loss_D.item()
                }
    
    def train(self):
        ret = super().train()
        return ret