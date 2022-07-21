# from distutils.command.config import config
"""
attention-guided. extract the attention map to highlight the important area
"""
# from pydantic import create_model
from cmath import e
from pyexpat import model
import torch.nn as nn
import timm
import torch
import numpy as np
# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
import cv2
from torchvision import transforms
# from code.models.cbam import ResNetCBAM
# from models.conformer import Conformer
# from models.sasa import ResNetSASA
# from models.sasa import Bottleneck as BNSASA
# from models.cbam import ResNetCBAM
# from models.cbam import Bottleneck as BNCBAM
# from models.sa import ResNetSA, SABottleneck
from models.se import SEResNet, Bottleneck

# def reshape_transform(tensor, height=7, width=7):
#     result = tensor.reshape(tensor.size(0),
#         height, width, tensor.size(2))
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result

# def fit_box(box_x, box_y, box_width, box_height):
    
#     if (box_height != 0) & (box_width != 0):
#         if box_width > box_height:
#             diff = box_width - box_height
#             if box_y > diff//2:
#                 box_y -= diff//2
#             box_height = box_width
#         elif box_width < box_height:
#             diff = box_height - box_width
#             if box_x > diff//2:
#                 box_x -= diff//2
#             box_width = box_height
#     else:
#         box_x, box_y = 0, 0
#         box_width, box_height = 224, 224
    
#     return box_x, box_y, box_width, box_height

# def extract_saliency_map(input_tensor, grayscale_cam, threshold, img_size):
#     grayscale_cam = np.moveaxis(grayscale_cam, 0, -1)
#     tf_resize = transforms.Resize((img_size, img_size))
#     binarized_saliencymaps = grayscale_cam > threshold
#     binarized_saliencymaps = binarized_saliencymaps*255
#     binarized_saliencymaps = binarized_saliencymaps.astype(np.uint8)
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
#     batch_size = input_tensor.shape[0]
#     cropped_tensors = torch.empty((batch_size, 3, img_size, img_size), device=device)
#     for _idx in range(binarized_saliencymaps.shape[-1]):
#         map = binarized_saliencymaps[:,:,_idx]
#         map = np.expand_dims(map, axis=-1)
#         box_x, box_y, box_width, box_height = cv2.boundingRect(map)
#         box_x, box_y, box_width, box_height = fit_box(box_x, box_y, box_width, box_height)

#         cropped_img = torch.unsqueeze(input_tensor[_idx,:,box_y:box_y+box_height, box_x:box_x+box_width],dim=0)
#         cropped_img = torch.squeeze(tf_resize(cropped_img))
#         cropped_tensors[_idx] = cropped_img
    
#     return cropped_tensors

# class AttentionGuideCNN(nn.Module):
#     def __init__(self,  img_size = 224,
#                         model_name = 'swin_tiny_patch4_window7_224',
#                         # num_features = 7,
#                         batch_size = 32,
#                         num_classes = 11,
#                         threshold = 0.3) -> None:
#         super().__init__()

#         self.img_size = img_size
#         self.model_swin = timm.create_model(model_name,num_classes)
#         self.target_layers = [self.model_swin.layers[-1].blocks[-1].norm1]
#         self.batch_size = batch_size
#         self.threshold = threshold
#     def forward(self,x, is_valid = False):
#         if is_valid:
#             x_global_cls = self.model_swin(x)
#             return x_global_cls
#         else:
#             cam = GradCAMPlusPlus(model=self.model_swin,
#                                 target_layers=self.target_layers,
#                                 reshape_transform=reshape_transform,
#                                 use_cuda=torch.cuda.is_available())
#             cam.batch_size = self.batch_size
#             grayscale_cam = cam(input_tensor=x)
#             x_crop = extract_saliency_map(input_tensor=x, 
#                                         grayscale_cam=grayscale_cam, 
#                                         threshold=self.threshold, 
#                                         img_size=self.img_size)
            
#             x_global_cls = self.model_swin(x)
#             x_local_cls = self.model_swin(x_crop)
#             return x_global_cls, x_local_cls

def build_head(in_fts, out_fts, is_complex = False):
    if is_complex:
        print('Build complex MLP head')
        head = nn.Sequential(
                        nn.Linear(in_fts, in_fts//4), 
                        nn.ReLU(), 
                        nn.Dropout(0.2), 
                        nn.BatchNorm1d(in_fts//4), 
                        nn.Linear(in_fts//4, out_fts)
                        )   
    else:
        print('Build simple MLP head')
        head = nn.Linear(in_fts, out_fts, bias = True)
    return head

class ModelMargin(nn.Module):
    def __init__(self, model_name, pretrained, num_classes):
        super().__init__()
        self.model = timm.create_model(model_name,
                                    pretrained=pretrained,
                                    num_classes = num_classes)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes, bias=False)
        self.backbone = nn.Sequential(*(list(self.model.children())[:-1]))
        self.fc = self.model.fc
    def forward(self, x):
        return self.model(x)

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class ModelwEmb(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, low_dim = 256, is_complex = False):
        super().__init__()
        ## load pre-trained weight abnormality classification
        self.model_name = model_name
        if model_name == 'resnet50se':
            self.model = SEResNet(block = Bottleneck, layers = [3, 4, 6, 3])
            in_fts = self.model.fc.in_features
            self.model.fc = build_head(in_fts, 2)

        # elif model_name == 'resnet50cbam':
        #     self.model = ResNetCBAM(BNCBAM, [3, 4, 6, 3], "ImageNet", 1000, "CBAM")
        #     in_fts = self.model.fc.in_features
        #     self.model.fc = build_head(in_fts, 2, True)
        # elif model_name == 'resnet50sa':
        #     self.model = ResNetSA(block = SABottleneck, layers = [3,4,6,3])
        #     in_fts = self.model.fc.in_features
        #     self.model.fc = build_head(in_fts, 2)
        else:
            if pretrained:
                self.model = timm.create_model(model_name, pretrained=False, num_classes = 2)
            else:
                self.model = timm.create_model(model_name, pretrained=True, num_classes = 2)

            if str(model_name).startswith('resnet'):
                in_fts = self.model.fc.in_features
            else:
                in_fts = self.model.classifier.in_features
        self.k = 3
        self.model_name = model_name
        if pretrained != 'None':
            print('Load checkpoint Abnormal')
            self.checkpoint = torch.load(pretrained, map_location = {'cuda:0':'cpu'})
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        ## transfer
        # if model_name == 'densenet161':
        #     in_fts = self.model.classifier.in_features
        #     # self.model.classifier = nn.Linear(in_fts, num_classes, bias=False)
        #     self.model.classifier = build_head(in_fts, num_classes)
        #     self.backbone = nn.Sequential(*(list(self.model.children())[:-1]))
        #     self.classifier = self.model.classifier
        # else:
        #     in_fts = self.model.fc.in_features
        #     # self.model.fc = nn.Linear(in_fts, num_classes, bias=False)
        #     self.model.fc = build_head(in_fts, num_classes)
        #     self.backbone = nn.Sequential(*(list(self.model.children())[:-1]))
        #     self.fc = self.model.fc
        if str(model_name).startswith('resnet'):
            self.model.fc = build_head(in_fts, num_classes, is_complex = True)
            self.fc = self.model.fc
        else:
            self.model.classifier = build_head(in_fts, num_classes, is_complex = True)
            self.fc = self.model.classifier
        self.backbone = nn.Sequential(*(list(self.model.children())[:-1]))
        self.head_emb = nn.Sequential(
            nn.Linear(in_fts, low_dim * self.k),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(low_dim * self.k, low_dim), 
            Normalize(2))

    def forward(self, x):
        fts = self.backbone(x)
        fts = torch.flatten(fts, 1)
        logits = self.fc(fts)

        fts_low = self.head_emb(fts)
        return logits, fts, fts_low