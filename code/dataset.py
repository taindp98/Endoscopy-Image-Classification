from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from randaugment import RandAugmentMC

import cv2
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import os
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class TransformFixMatch(object):
    def __init__(self, config, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize((config['size'],config['size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=config['size'],
                                  padding=int(config['size']*0.125),
                                  padding_mode='reflect')])
        
        self.strong = transforms.Compose([
            transforms.Resize((config['size'],config['size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=config['size'],
                                  padding=int(config['size']*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def get_transform(config, is_train = False, is_labeled = True):
    if is_train:
        if is_labeled:
            trf_aug = transforms.Compose([
                transforms.Resize((config['size'],config['size'])),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            trf_aug = TransformFixMatch(config, mean, std)
    else:
        trf_aug = transforms.Compose([
            transforms.Resize((config['size'],config['size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    return trf_aug

class GIDataset(torch.utils.data.Dataset):
    def __init__(self, df, config, transforms=None):
        self.df = df
        self.transforms = transforms
        self.len = df.shape[0]
        self.config = config

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row['Image']
        x = cv2.imread(os.path.join(self.config['data_path'], img_name))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = Image.fromarray(x)
        vec = np.array(row['Groupby_Categories'], dtype=float)
        y = torch.tensor(vec, dtype=torch.long)
        
        if self.transforms:
            x = self.transforms(x)
        return x, y