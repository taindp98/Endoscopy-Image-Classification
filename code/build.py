# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

# from models.swin_transformer import SwinTransformer
# from models.swin_mlp import SwinMLP
# from models.coat_net import CoAtNet
from models.custom_model import ModelMargin, ModelCoMatch
from pydantic import create_model
import torch.nn as nn
from torch.nn import DataParallel
import timm
import torch
from torch import nn
from models.conformer import Conformer
from torchvision import models
from fastai.layers import PooledSelfAttention2d


def build_head(in_fts, out_fts):
    head = nn.Sequential(
                        # PooledSelfAttention2d(in_fts), 
                        # nn.AdaptiveAvgPool2d(1), 
                        # nn.Flatten(),
                        # nn.Dropout(0.5), 
                        nn.Linear(in_fts, in_fts//4), 
                        nn.ReLU(), 
                        nn.Dropout(0.25), 
                        nn.BatchNorm1d(in_fts//4), 
                        nn.Linear(in_fts//4, out_fts)
                        )   

    return head


def build_model(config, is_pathology = True):
    model_name = config.MODEL.NAME
    if model_name == 'swin':
        model = timm.create_model('swin_base_patch4_window7_224', 
                                pretrained = config.MODEL.PRE_TRAIN, 
                                num_classes = config.MODEL.NUM_CLASSES)
        # model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
        #                         num_classes=config.MODEL.NUM_CLASSES,
        #                         patch_size= 4,
        #                         in_chans= 3,
        #                         embed_dim= 96,
        #                         depths= [2, 2, 6, 2],
        #                         num_heads= [3, 6, 12, 24],
        #                         window_size= 7,
        #                         mlp_ratio= 4.,
        #                         qkv_bias= True,
        #                         qk_scale= None,
        #                         drop_rate= 0.,
        #                         drop_path_rate= 0.1,
        #                         ape= False,
        #                         patch_norm= True,
        #                         use_checkpoint=None)
    elif model_name == 'swin_mlp':
        model = timm.create_model('swin_mlp_base_patch4_window7_224', 
                                pretrained = config.MODEL.PRE_TRAIN, 
                                num_classes = config.MODEL.NUM_CLASSES)
        # model = SwinMLP(img_size=config.DATA.IMG_SIZE,
        #                 num_classes=config.MODEL.NUM_CLASSES,
        #                 patch_size= 4,
        #                 in_chans= 3,
        #                 embed_dim= 96,
        #                 depths= [2, 2, 6, 2],
        #                 num_heads= [3, 6, 12, 24],
        #                 window_size= 7,
        #                 mlp_ratio= 4.,
        #                 drop_rate= 0.,
        #                 drop_path_rate= 0.1,
        #                 ape= False,
        #                 patch_norm= True,
        #                 use_checkpoint=None)
        
    # elif model_name == 'coatnet':
    #     model = CoAtNet(image_size = (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), 
    #                     num_classes=config.MODEL.NUM_CLASSES, 
    #                     # batch_size = config.DATA.BATCH_SIZE,
    #                     in_channels = 3, 
    #                     num_blocks = [2, 2, 3, 5, 2], 
    #                     channels = [64, 96, 192, 384, 768], 
    #                     block_types=['C', 'C', 'T', 'T'])
    
    elif model_name == 'conformer':
        if config.MODEL.PRE_TRAIN:
            model = Conformer(patch_size=16, 
                        num_classes = 1000,
                        channel_ratio=1, 
                        embed_dim=384, 
                        depth=12,
                        num_heads=6, 
                        mlp_ratio=4, 
                        qkv_bias=True)

            

            checkpoint = torch.load(config.MODEL.PRE_TRAIN_PATH, map_location = {'cuda:0':'cpu'})
            if is_pathology:
                num_ftrs_conv = model.conv_cls_head.in_features
                num_ftrs_trans = model.trans_cls_head.in_features
                model.conv_cls_head = nn.Linear(num_ftrs_conv, 2)
                model.trans_cls_head = nn.Linear(num_ftrs_trans, 2)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            num_ftrs_conv = model.conv_cls_head.in_features
            num_ftrs_trans = model.trans_cls_head.in_features
            model.conv_cls_head = build_head(num_ftrs_conv, config.MODEL.NUM_CLASSES)
            model.trans_cls_head = build_head(num_ftrs_trans, config.MODEL.NUM_CLASSES)
            # model.conv_cls_head = nn.Linear(num_ftrs_conv, config.MODEL.NUM_CLASSES)
            # model.trans_cls_head = nn.Linear(num_ftrs_trans, config.MODEL.NUM_CLASSES)
        else:
            model = Conformer(patch_size=16, 
                        num_classes = config.MODEL.NUM_CLASSES,
                        channel_ratio=1, 
                        embed_dim=384, 
                        depth=12,
                        num_heads=6, 
                        mlp_ratio=4, 
                        qkv_bias=True)
    elif config.MODEL.TYPE_SEMI == 'CoMatch':
        model = ModelCoMatch(model_name, pretrained = config.MODEL.PRE_TRAIN_PATH, num_classes= config.MODEL.NUM_CLASSES)
    else:
        if config.MODEL.MARGIN != 'None':
            model = ModelMargin(model_name, pretrained=True, num_classes=config.MODEL.NUM_CLASSES)

        else:
            if config.MODEL.PRE_TRAIN_PATH != 'None':
                model = timm.create_model(model_name, pretrained=True, num_classes = 2)
                checkpoint = torch.load(config.MODEL.PRE_TRAIN_PATH, map_location = {'cuda:0':'cpu'})
                model.load_state_dict(checkpoint['model_state_dict'])
                print('Loaded checkpoint abnormal')
                in_fts = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_features= in_fts, out_features= config.MODEL.NUM_CLASSES, bias= True)
            else:
                model = timm.create_model(model_name, pretrained=True, num_classes = config.MODEL.NUM_CLASSES)

    return model