from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from datetime import datetime,date
import os
import yaml
from yaml.loader import SafeLoader
import cv2

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_metrics(pred, target, config):
    # pred = np.argmax(pred, axis=1)
#     target = np.argmax(target, axis=1)
    res = []
    for l in range(config.MODEL.NUM_CLASSES):
        prec, recall, _, _ = precision_recall_fscore_support(np.array(target)==l, np.array(pred)==l, pos_label=True,average=None)
        res.append([l, recall[1], recall[0]])

    df_sen_spec = pd.DataFrame(res,columns = ['class','sensitivity','specificity'])

    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'sen/spec': df_sen_spec
           }
    


def show_cfs_matrix(targ, pred):
    C = confusion_matrix(targ, pred)
    cmn = C / C.astype('float').sum(axis=1)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    
def show_batch(inp, title=None):
    """Imshow for Tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig=plt.figure(figsize=(20, 7))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_imgs(list_fnames):
    fig=plt.figure(figsize=(20,7))
    rows = 2
    columns = len(list_fnames)//rows
    k = 0
    for i in range(0, rows):
        for j in range(0, columns):
            img = cv2.imread(list_fnames[k])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            fig.add_subplot(rows, columns, k+1)
            plt.imshow(img)
            plt.axis('off')
            k += 1

def show_grid(list_imgs):
    fig=plt.figure(figsize=(20,7))
    rows = 1
    columns = len(list_imgs)//rows
    k = 0
    for i in range(0, rows):
        for j in range(0, columns):
            # img = cv2.imread(list_imgs[k])
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inp = list_imgs[k]
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
    
            fig.add_subplot(rows, columns, k+1)
            plt.imshow(inp)
            plt.axis('off')
            k += 1
### UTILS FIXMATCH
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def get_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=SafeLoader)
    config = AttrDict(config)
    for k1 in config.keys():
        config[k1] = AttrDict(config[k1])
    return config

def resize_aspect_ratio(img, size, interp=cv2.INTER_LINEAR):
    """
    resize min edge to target size, keeping aspect ratio
    """
    if len(img.shape) == 2:
        h,w = img.shape
    elif len(img.shape) == 3:
        h,w,_ = img.shape
    else:
        return None
    if h > w:
        new_w = size
        new_h = h*new_w//w
    else:
        new_h = size
        new_w = w*new_h//h
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)