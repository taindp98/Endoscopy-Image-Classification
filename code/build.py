# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from models.swin_transformer import SwinTransformer
from models.swin_mlp import SwinMLP
from models.coat_net import CoAtNet
from models.custom_model import AttentionGuideCNN
import torch.nn as nn
from torch.nn import DataParallel
from utils import load_checkpoint
import timm


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = timm.create_model('swin_base_patch4_window7_224', 
                                pretrained = True, 
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
    elif model_type == 'swin_mlp':
        model = timm.create_model('swin_mlp_base_patch4_window7_224', 
                                pretrained = True, 
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
        
    elif model_type == 'coat':
        model = CoAtNet(image_size = (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), 
                        num_classes=config.MODEL.NUM_CLASSES, 
                        batch_size = config.DATA.BATCH_SIZE,
                        in_channels = 3, 
                        num_blocks = [2, 2, 3, 5, 2], 
                        channels = [64, 96, 192, 384, 768], 
                        block_types=['C', 'C', 'T', 'T'])
    elif model_type == 'attguide':
        model = AttentionGuideCNN(img_size = config.DATA.IMG_SIZE, 
                                num_classes=config.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    # model = DataParallel(model)
    model.to(str(config.TRAIN.DEVICE))
    if config.TRAIN._CHECKPOINT_MODEL:
        model = load_checkpoint(model, config.TRAIN._CHECKPOINT_MODEL, is_train = config.TRAIN.IS_TRAIN)    
    return model