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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import show_grid, show_batch

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class TransformFixMatch(object):
    def __init__(self, config, mean, std):
        self.weak = transforms.Compose([
            # transforms.CenterCrop(config.DATA.IMG_SIZE),
            transforms.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=config.DATA.IMG_SIZE,
                                  padding=int(config.DATA.IMG_SIZE*0.125),
                                  padding_mode='reflect')])
        
        self.strong = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=config.DATA.IMG_SIZE,
                                  padding=int(config.DATA.IMG_SIZE*0.125),
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
                # transforms.CenterCrop(config.DATA.IMG_SIZE),
                transforms.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                # transforms.RandomRotation(20),
                transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.63, 1)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            trf_aug = TransformFixMatch(config, mean, std)
    else:
        trf_aug = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    return trf_aug

class GIDataset(torch.utils.data.Dataset):
    def __init__(self, df, config, transforms=None, is_unanno = False):
        self.df = df
        self.transforms = transforms
        self.len = df.shape[0]
        self.config = config
        self.input_name = config.DATA.INPUT_NAME
        self.target_name = config.DATA.TARGET_NAME
        self.is_unanno = is_unanno
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row[self.input_name]
        # if self.config.TRAIN.IS_SSL:
        #     if self.config.DATA.MOCKUP_SSL:
                # x = cv2.imread(os.path.join(self.config.DATA.PATH, img_name))
                # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                # x = Image.fromarray(x)
                # if self.transforms:
                #     x = self.transforms(x)
                # vec = np.array(row[self.target_name], dtype=float)
                # y = torch.tensor(vec, dtype=torch.long)
            # else:
        if self.is_unanno:
            if self.config.DATA.MOCKUP_SSL:
                data_dir = self.config.DATA.PATH
            else:
                data_dir = self.config.DATA.UNANNO_PATH
            x = cv2.imread(os.path.join(data_dir, img_name))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = Image.fromarray(x)
            if self.transforms:
                x = self.transforms(x)
            return x
        else:
            x = cv2.imread(os.path.join(self.config.DATA.PATH, img_name))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = Image.fromarray(x)
            if self.transforms:
                x = self.transforms(x)
            vec = np.array(row[self.target_name], dtype=float)
            y = torch.tensor(vec, dtype=torch.long)
        # else:
        #     x = cv2.imread(os.path.join(self.config.DATA.PATH, img_name))
        #     x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        #     x = Image.fromarray(x)
        #     if self.transforms:
        #         x = self.transforms(x)
        #     vec = np.array(row[self.target_name], dtype=float)
        #     y = torch.tensor(vec, dtype=torch.long)
            return x, y

def get_data(config, df_anno, df_unanno = None, is_visual=False):
    """
    get training, validation, testing set
    """
    df_train = df_anno[df_anno['is_valid']==False]
    df_valid = df_anno[df_anno['is_valid']==True]
    ## break down into labeled and unlabeled set
    if config.TRAIN.IS_SSL:
        if config.DATA.MOCKUP_SSL:
            df_labeled, df_unlabeled = train_test_split(df_train, test_size = config.DATA.MOCKUP_SIZE, random_state = 0)
            train_labeled_ds = GIDataset(df = df_labeled, config = config, transforms = get_transform(config, is_train=True))
            train_unlabeled_ds = GIDataset(df = df_unlabeled, config = config, transforms = get_transform(config, is_train=True, is_labeled=False), is_unanno = True)
            train_labeled_dl = DataLoader(train_labeled_ds, 
                                        sampler=RandomSampler(train_labeled_ds),
                                        batch_size = config.DATA.BATCH_SIZE, 
                                        num_workers = config.DATA.NUM_WORKERS)

            train_unlabeled_dl = DataLoader(train_unlabeled_ds, 
                                        sampler=RandomSampler(train_unlabeled_ds),
                                        batch_size = config.DATA.BATCH_SIZE*config.DATA.MU, 
                                        num_workers = config.DATA.NUM_WORKERS)
            train_dl = (train_labeled_dl, train_unlabeled_dl)

            if is_visual:
                for x1, y1 in train_labeled_dl:
                    # print(x.shape)
                    # print(y)
                    break
                for x2 in train_unlabeled_dl:
                    break
                show_grid([x1[0,:,:], x2[0][0,:,:], x2[1][0,:,:]])
        ## else for real unlabeled data
        else:
            train_labeled_ds = GIDataset(df = df_train, config = config, transforms = get_transform(config, is_train=True))
            train_unlabeled_ds = GIDataset(df = df_unanno, config = config, transforms = get_transform(config, is_train=True, is_labeled=False), is_unanno = True)
            train_labeled_dl = DataLoader(train_labeled_ds, 
                                        sampler=RandomSampler(train_labeled_ds),
                                        batch_size = config.DATA.BATCH_SIZE, 
                                        num_workers = config.DATA.NUM_WORKERS)

            train_unlabeled_dl = DataLoader(train_unlabeled_ds, 
                                        sampler=RandomSampler(train_unlabeled_ds),
                                        batch_size = config.DATA.BATCH_SIZE*config.DATA.MU, 
                                        num_workers = config.DATA.NUM_WORKERS)
            train_dl = (train_labeled_dl, train_unlabeled_dl)
            if is_visual:
                for x1, y1 in train_labeled_dl:
                    # print(x.shape)
                    # print(y)
                    break
                for x2, y2 in train_unlabeled_dl:
                    break
                show_grid([x1[0,:,:], x2[0][0,:,:], x2[1][0,:,:]])
    else:
        train_ds = GIDataset(df_train, config = config, transforms = get_transform(config, is_train=True))
        train_dl = DataLoader(train_ds, 
                        sampler=RandomSampler(train_ds),
                        batch_size = config.DATA.BATCH_SIZE, 
                        num_workers = config.DATA.NUM_WORKERS)
        if is_visual:
            for x, y in train_dl:
                # print(x.shape)
                # print(y)
                break
            show_batch(x[0,:,:], y[0])
    valid_ds = GIDataset(df_valid , config = config, transforms = get_transform(config))
    valid_dl = DataLoader(valid_ds, 
                        sampler=SequentialSampler(valid_ds),
                        batch_size = config.DATA.BATCH_SIZE, 
                        num_workers = config.DATA.NUM_WORKERS)

    return train_dl, valid_dl