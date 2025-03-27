import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


class GANdataloader(torch.utils.data.Dataset):
    def __init__(self,train=True, batch_size = 64):
        """"
        Initialize the dataloader class.

        Args:
            train : whether to use training dataset or test dataset (type : bool)
            batch_size : how many samples per batch to load (type : int)
        """
        # super(GANdataloader, self).__init__()
        super().__init__()

        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
   
        data = np.load("./data/sprites_1788_16x16.npy")
        label = np.load("./data/sprite_labels_nc_1788_16x16.npy")
        
        data = np.transpose(data, (0, 3, 1, 2))
        assert data.shape == (89400, 3, 16, 16), "Incorrect dataset shape"

        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        self.dataset = torch.utils.data.TensorDataset(data, label)
        shuffle = train  
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __getitem__(self, idx):
        return self.dataloader[idx]
