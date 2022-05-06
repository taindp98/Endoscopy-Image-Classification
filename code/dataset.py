import cv2
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

def deformation(img):
    h, w = img.shape[:2]
    xcent = w / 2
    ycent = h / 2

    # set up the maps as float32 from output square (x,y) to input circle (u,v)
    map_u = np.zeros((h, w), np.float32)
    map_v = np.zeros((h, w), np.float32)

    # create u and v maps where x,y is measured from the center and scaled from -1 to 1
    for y in range(h):
        y_norm = (y - ycent)/ycent
        for x in range(w):
            x_norm = (x - xcent)/xcent
            map_u[y, x] = xcent * x_norm * np.sqrt(1 - 0.5*y_norm**2) + xcent
            map_v[y, x] = ycent * y_norm * np.sqrt(1 - 0.5*x_norm**2) + ycent
    img_deform = cv2.remap(img, map_u, map_v, cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT_101, borderValue=(0,0,0))
    return img_deform

class GIDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, config, transforms=None):
        self.data_frame = data_frame
        self.transforms = transforms
        self.len = data_frame.shape[0]
        self.config = config

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        img_path = row['path']
        x = cv2.imread(img_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#         x = deformation(x)
        x = Image.fromarray(x)
        vec = np.array(row['target'], dtype=float)
        y = torch.tensor(vec, dtype=torch.long)
#         y = F.one_hot(y, num_classes = self.config.MODEL.NUM_CLASSES)
#         y = y.type(torch.FloatTensor)
        
        if self.transforms:
            x = self.transforms(x)
        return x, y