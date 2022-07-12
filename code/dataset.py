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
from timm.data import Mixup
import random

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class TransformFixMatch(object):
    def __init__(self, config, mean, std):
        if config.DATA.IS_CROP:
            self.weak = transforms.Compose([
                    transforms.Resize((int(config.DATA.IMG_SIZE*1.2),int(config.DATA.IMG_SIZE*1.2))),
                    transforms.CenterCrop(config.DATA.IMG_SIZE)])
                    
            self.strong = transforms.Compose([
                transforms.Resize((int(config.DATA.IMG_SIZE*1.2),int(config.DATA.IMG_SIZE*1.2))),    
                transforms.CenterCrop(config.DATA.IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=config.DATA.IMG_SIZE,
                                    padding=int(config.DATA.IMG_SIZE*0.125),
                                    padding_mode='reflect'),
                RandAugmentMC(n=2, m=10)])
        else:
            self.weak = transforms.Compose([
                transforms.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE))])
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

class TransformCoMatch(object):
    def __init__(self, config, mean, std):
        if config.DATA.IS_CROP:
            self.weak = transforms.Compose([
                    transforms.Resize((int(config.DATA.IMG_SIZE*1.2),int(config.DATA.IMG_SIZE*1.2))),
                    transforms.CenterCrop(config.DATA.IMG_SIZE),
                    transforms.RandomHorizontalFlip()])
                    
            self.strong_0 = transforms.Compose([
                transforms.Resize((int(config.DATA.IMG_SIZE*1.2),int(config.DATA.IMG_SIZE*1.2))),    
                transforms.CenterCrop(config.DATA.IMG_SIZE),                      
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=2, m=10)
                ])

            self.strong_1 = transforms.Compose([
                transforms.Resize((int(config.DATA.IMG_SIZE*1.2),int(config.DATA.IMG_SIZE*1.2))),    
                transforms.CenterCrop(config.DATA.IMG_SIZE),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) 
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),                         
                transforms.RandomHorizontalFlip()
                ])
        else:
            self.weak = transforms.Compose([
                transforms.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)),
                transforms.RandomHorizontalFlip()])
            
            
            self.strong_0 = transforms.Compose([
                transforms.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)),                       
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=2, m=10)
                ])

            self.strong_1 = transforms.Compose([
                transforms.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),                         
                transforms.RandomHorizontalFlip()
                ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong_0 = self.strong_0(x)
        strong_1 = self.strong_1(x)
        return self.normalize(weak), self.normalize(strong_0), self.normalize(strong_1)

def reproduce_transform(is_train = False):
    if is_train:
        print('Transforms mode: Re-produce paper')
        trf_aug = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        trf_aug = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    return trf_aug


class EmbFeatEZBM(Dataset):
    def __init__(self, data, target, cls_num_list):
        self.data = data
        self.target = target
        self.class_dict = dict()
        self.cls_num_list = cls_num_list
        self.cls_num = len(cls_num_list)
        self.type = 'balance'
        for i in range(self.cls_num):
            idx = torch.where(self.target == i)[0]
            self.class_dict[i] = idx

        # prob for reverse
        cls_num_list = np.array(self.cls_num_list)
        prob = list(cls_num_list / np.sum(cls_num_list))
        prob.reverse()
        self.prob = np.array(prob)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        if self.type == 'balance':
            sample_class = random.randint(0, self.cls_num - 1)
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)

        if self.type == 'reverse':
            sample_class = np.random.choice(range(self.cls_num), p=self.prob.ravel())
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)

        temp_class = random.randint(0, self.cls_num - 1)
        temp_indexes = self.class_dict[temp_class]
        temp_index = random.choice(temp_indexes)
        item = temp_index

        data, target = self.data[item], self.target[item]
        data_dual, target_dual = self.data[sample_index], self.target[sample_index]

        return data, target, data_dual, target_dual

def get_transform(config, is_train = False, is_labeled = True, type_semi = 'FixMatch', is_reprod = False):
    if is_reprod:
        
        trf_aug = reproduce_transform(is_train = is_train)
    else:
        if is_train:
            if is_labeled:
                if config.DATA.IS_CROP:
                    trf_aug = transforms.Compose([
                        # transforms.CenterCrop(config.DATA.IMG_SIZE),
                        transforms.Resize((int(config.DATA.IMG_SIZE*1.2),int(config.DATA.IMG_SIZE*1.2))),
                        # transforms.Resize((272,272)),
                        transforms.RandomHorizontalFlip(p=0.3),
                        transforms.RandomVerticalFlip(p=0.3),
                        transforms.RandomRotation(20),
                        transforms.CenterCrop(config.DATA.IMG_SIZE),
                        transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
                        # transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.63, 1)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)])
                else:
                    trf_aug = transforms.Compose([
                        transforms.Resize((int(config.DATA.IMG_SIZE),int(config.DATA.IMG_SIZE))),
                        transforms.RandomHorizontalFlip(p=0.3),
                        transforms.RandomVerticalFlip(p=0.3),
                        transforms.RandomRotation(20),
                        transforms.CenterCrop(config.DATA.IMG_SIZE),
                        transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
                        # transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.63, 1)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)])
            else:
                if type_semi in ['FixMatch', 'SemiFormer']:
                    print(f'Semi Transform mode: {type_semi}')
                    trf_aug = TransformFixMatch(config, mean, std)
                else:
                    print('Semi Transform mode: CoMatch')
                    trf_aug = TransformCoMatch(config, mean, std)

        else:
            # print('Validation transform')
            if config.DATA.IS_CROP:
                trf_aug = transforms.Compose([
                    # transforms.Resize((272,272)),
                    transforms.Resize((int(config.DATA.IMG_SIZE*1.2),int(config.DATA.IMG_SIZE*1.2))),
                    transforms.CenterCrop(config.DATA.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
            else:
                trf_aug = transforms.Compose([
                    # transforms.Resize((272,272)),
                    transforms.Resize((int(config.DATA.IMG_SIZE),int(config.DATA.IMG_SIZE))),
                    transforms.CenterCrop(config.DATA.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
    return trf_aug

class GIDataset(torch.utils.data.Dataset):
    def __init__(self, df, config, transforms=None, is_unanno = False, is_triplet = False):
        self.df = df
        self.transforms = transforms
        self.len = df.shape[0]
        self.config = config
        self.input_name = config.DATA.INPUT_NAME
        self.target_name = config.DATA.TARGET_NAME
        self.is_unanno = is_unanno
        self.is_triplet = is_triplet

    def __len__(self):
        return self.len

    def get_labeled_data(self, img_name, target):
        x = cv2.imread(os.path.join(self.config.DATA.PATH, img_name))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = Image.fromarray(x)
        if self.transforms:
            x = self.transforms(x)
        y = torch.tensor(target, dtype=torch.long)
        return x,y

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.config.MODEL.NUM_CLASSES):
            num_ex = self.df[self.df['target']==i]
            cls_num_list.append(len(num_ex))
        return cls_num_list

    def __getitem__(self, index):
        
        img_name = self.df.iloc[index][self.input_name]
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
            return x, index
        else:
            if self.is_triplet:
                list_unique_cls = list(set(list(self.df[self.target_name])))
                anchor_img_name = self.df.iloc[index][self.input_name]
                anchor_cls = self.df.iloc[index][self.target_name]

                pos_img_name = anchor_img_name
                pos_cls = None

                neg_img_name = None
                neg_cls = anchor_cls

                while (pos_img_name == anchor_img_name) | (pos_cls != anchor_cls):
                    df_pos = self.df.sample(1).iloc[0]
                    pos_img_name = df_pos[self.input_name]
                    pos_cls = df_pos[self.target_name]

                while (neg_img_name == anchor_img_name) | (neg_cls == anchor_cls):
                    df_neg = self.df.sample(1).iloc[0]
                    neg_img_name = df_neg[self.input_name]
                    neg_cls = df_neg[self.target_name]


                anchor_x, anchor_y = self.get_labeled_data(img_name = anchor_img_name, target = anchor_cls)
                pos_x, pos_y = self.get_labeled_data(img_name = pos_img_name, target = pos_cls)
                neg_x, neg_y = self.get_labeled_data(img_name = neg_img_name, target = neg_cls)

                x = tuple([anchor_x, pos_x, neg_x])
                y = tuple([anchor_y, pos_y, neg_y])

            else:
                vec = np.array(self.df.iloc[index][self.target_name], dtype=float)
                x, y = self.get_labeled_data(img_name = img_name, target = vec)

            return x, y

def get_data(config, df_anno, df_unanno = None, is_full_sup = True, is_visual=False, type_semi = 'FixMatch', predict = False, is_reprod = False):
    """
    get training, validation, testing set
    """
    df_train = df_anno[df_anno['is_valid']==False]
    df_valid = df_anno[df_anno['is_valid']==True]
    ## break down into labeled and unlabeled set

    mixup_fn = None
    if config.TRAIN.CUTMIX_MINMAX == 'None':
        config.TRAIN.CUTMIX_MINMAX = None
    mixup_active = config.TRAIN.MIXUP > 0. or config.TRAIN.CUTMIX > 0. or config.TRAIN.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.TRAIN.MIXUP, cutmix_alpha=config.TRAIN.CUTMIX, cutmix_minmax=config.TRAIN.CUTMIX_MINMAX,
            prob=config.TRAIN.MIXUP_PROB, switch_prob=config.TRAIN.MIXUP_SWITCH_PROB, mode=config.TRAIN.MIXUP_MODE,
            label_smoothing=config.TRAIN.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    if predict:
        unlabeled_ds = GIDataset(df = df_unanno, config = config, transforms = get_transform(config = config, is_train=False, is_labeled=False, type_semi = type_semi, is_reprod = is_reprod), is_unanno = True)
        
        unlabeled_dl = DataLoader(unlabeled_ds, 
                                    sampler=RandomSampler(unlabeled_ds),
                                    batch_size = config.DATA.BATCH_SIZE, 
                                    num_workers = config.DATA.NUM_WORKERS)

        return unlabeled_dl
    else:
        if not is_full_sup:
            if config.TRAIN.IS_SSL:
                if config.DATA.MOCKUP_SSL:
                    print('Training mode: Mock labeled and unlabeled data')
                    df_labeled = df_train[df_train['is_labeled']==True]
                    df_unlabeled = df_train[df_train['is_labeled']==False]
                    train_labeled_ds = GIDataset(df = df_labeled, config = config, transforms = get_transform(config, is_train=True), is_triplet = config.MODEL.IS_TRIPLET)
                    train_unlabeled_ds = GIDataset(df = df_unlabeled, config = config, transforms = get_transform(config, is_train=True, is_labeled=False, type_semi = type_semi), is_unanno = True)
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
                        for x2, _ in train_unlabeled_dl:
                            break
                        show_grid([x1[0,:,:], x2[0][0,:,:], x2[1][0,:,:]])
                ## else for real unlabeled data
                else:
                    print('Training mode: Full labeled and unlabeled data')
                    train_labeled_ds = GIDataset(df = df_train, config = config, transforms = get_transform(config, is_train=True), is_triplet = config.MODEL.IS_TRIPLET)

                    df_unlabeled = df_unanno[df_unanno['pred']==1]
                    train_unlabeled_ds = GIDataset(df = df_unlabeled, 
                                                    config = config, 
                                                    transforms = get_transform(config, is_train=True, is_labeled=False, type_semi = type_semi), 
                                                    is_unanno = True)
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
                            break
                        for x2, _ in train_unlabeled_dl:
                            break
                        if type_semi in ['FixMatch', 'SemiFormer']:
                            show_grid([x1[0,:,:], x2[0][0,:,:], x2[1][0,:,:]])
                        else:
                            show_grid([x1[0,:,:], x2[0][0,:,:], x2[1][0,:,:], x2[2][0,:,:]])

            else:
                train_ds = GIDataset(df_train[df_train['is_labeled']==True], config = config, transforms = get_transform(config = config, is_train=True, is_reprod = is_reprod), is_triplet = config.MODEL.IS_TRIPLET)
                train_dl = DataLoader(train_ds, 
                                sampler=RandomSampler(train_ds),
                                batch_size = config.DATA.BATCH_SIZE, 
                                num_workers = config.DATA.NUM_WORKERS)

            
            valid_ds = GIDataset(df_valid , config = config, transforms = get_transform(config = config, is_reprod = is_reprod))
            valid_dl = DataLoader(valid_ds, 
                                sampler=SequentialSampler(valid_ds),
                                batch_size = config.DATA.BATCH_SIZE, 
                                num_workers = config.DATA.NUM_WORKERS)

        else:
            print('Training mode: Full labeled supervised learning')
            train_ds = GIDataset(df_train, config = config, transforms = get_transform(config = config, is_train=True, is_reprod = is_reprod), is_triplet = config.MODEL.IS_TRIPLET)
            train_dl = DataLoader(train_ds, 
                            sampler=RandomSampler(train_ds),
                            batch_size = config.DATA.BATCH_SIZE, 
                            num_workers = config.DATA.NUM_WORKERS)
            

            valid_ds = GIDataset(df_valid , config = config, transforms = get_transform(config = config, is_reprod = is_reprod))
            valid_dl = DataLoader(valid_ds, 
                                sampler=SequentialSampler(valid_ds),
                                batch_size = config.DATA.BATCH_SIZE, 
                                num_workers = config.DATA.NUM_WORKERS)
            if is_visual:
                for x, y in train_dl:
                    break

                for x_vl, y_vl in valid_dl:
                    break
                ## if triplet show 3, else show 4
                if config.MODEL.IS_TRIPLET:
                    #x[0] is batch anchor, x[1] is batch pos, x[2] is batch neg
                    show_grid([x[0][0,:,:], x[1][0,:,:], x[2][0,:,:]])
                else:
                    show_grid([x[0,:,:], x[1,:,:], x[2,:,:], x[3,:,:]])
                    if mixup_fn is not None:
                        x,y = mixup_fn(x,y)
                    show_grid([x[0,:,:], x[1,:,:], x[2,:,:], x[3,:,:]])
                # show_grid([x_vl[0,:,:], x_vl[1,:,:], x_vl[2,:,:], x_vl[3,:,:]])    

    return train_dl, valid_dl, mixup_fn

