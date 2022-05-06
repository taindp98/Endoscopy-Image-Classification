# from distutils.command.config import config
import torch.nn as nn
import timm
import torch
import numpy as np
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from preprocess import reshape_transform, extract_saliency_map
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad


class AttentionGuideCNN(nn.Module):
    def __init__(self,  img_size = 224,
                        model_name = 'swin_tiny_patch4_window7_224',
                        # num_features = 7,
                        batch_size = 32,
                        num_classes = 11,
                        threshold = 0.3) -> None:
        super().__init__()

        self.img_size = img_size
        self.model_swin = timm.create_model(model_name,num_classes)
        self.target_layers = [self.model_swin.layers[-1].blocks[-1].norm1]
        self.batch_size = batch_size
        self.threshold = threshold
    def forward(self,x, is_valid = False):
        if is_valid:
            x_global_cls = self.model_swin(x)
            return x_global_cls
        else:
            cam = GradCAMPlusPlus(model=self.model_swin,
                                target_layers=self.target_layers,
                                reshape_transform=reshape_transform,
                                use_cuda=torch.cuda.is_available())
            cam.batch_size = self.batch_size
            grayscale_cam = cam(input_tensor=x)
            x_crop = extract_saliency_map(input_tensor=x, 
                                        grayscale_cam=grayscale_cam, 
                                        threshold=self.threshold, 
                                        img_size=self.img_size)
            
            x_global_cls = self.model_swin(x)
            x_local_cls = self.model_swin(x_crop)
            return x_global_cls, x_local_cls