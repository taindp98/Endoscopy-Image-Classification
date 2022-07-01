from fastai.vision import *
from fastai.callbacks.hooks import *
from PIL import Image
import os
from glob import glob
import cv2
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from matplotlib import pyplot as plt
from torchvision import transforms
import timm

import torch.nn.functional as F

from utils import show_batch, AverageMeter, show_grid, get_config
from fixmatch import FixMatch
from comatch import CoMatch
from supervised import SupLearning
from semiformer import SemiFormer
from dataset import get_data
from build import build_model

list_configs = ['./configs/kaggle_semisupervised_real_3_1.yaml', './configs/kaggle_semisupervised_real_3.yaml']


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

for idx, config in enumerate(list_configs):

    config = get_config(config)
    df_anno = pd.read_csv(config['DATA']['ANNO'])
    df_unanno = pd.read_csv(config['DATA']['UNANNO'])

    ## stage 1
    train_dl, valid_dl = get_data(config, 
                                df_anno, 
                                df_unanno, 
                                is_full_sup = False, 
                                is_visual=True, 
                                type_semi = config.MODEL.TYPE_SEMI)
    if idx == 0:
        model = build_model(config, is_pathology = True)

    if config.TRAIN.IS_SSL:
        if config.MODEL.TYPE_SEMI == 'FixMatch':
            classifier = FixMatch(model = model,
                            opt_func=config['TRAIN']['OPT_NAME'], 
                            device = device)
        elif config.MODEL.TYPE_SEMI == 'CoMatch':
            classifier = CoMatch(model = model,
                            opt_func=config['TRAIN']['OPT_NAME'], 
                            device = device)
        elif config.MODEL.TYPE_SEMI == 'SemiFormer':
            classifier = SemiFormer(model = model,
                            opt_func=config['TRAIN']['OPT_NAME'], 
                            device = device)
    else:
        classifier = SupLearning(model = model,
                            opt_func=config['TRAIN']['OPT_NAME'], 
                            device = device)
    
    classifier.get_dataloader(train_dl, valid_dl)
    classifier.get_config(config)

    classifier.fit()