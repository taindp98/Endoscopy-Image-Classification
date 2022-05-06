import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, class_weights = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        if class_weights != None:
            self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class AngularPenaltySMLoss(nn.Module):

    def __init__(self, config, s=None, m=None, eps=1e-7, weight = None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        # loss_type = loss_type.lower()
        # assert loss_type in  ['arcface', 'sphereface', 'cosface']

        self.loss_type = config.TRAIN.MARGIN
        if self.loss_type == 'arcface':
            self.s = 30.0 if not s else s
            self.m = 0.3 if not m else m
        if self.loss_type == 'sphereface':
            self.s = 30.0 if not s else s
            self.m = 1.35 if not m else m
        if self.loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.eps = eps
        self.weight = weight
        # self.bn = nn.BatchNorm1d(config.MODEL.NUM_CLASSES)
        self.device = str(config.TRAIN.DEVICE)

    def forward(self, input, target):
        '''
        input shape (N, in_features)
        '''
        assert len(input) == len(target)
        assert torch.min(target) >= 0
        
        # input = self.bn(input)
        input = F.normalize(input, p=2, dim=1)

        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(input.transpose(0, 1)[target]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(input.transpose(0, 1)[target]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(input.transpose(0, 1)[target]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((input[i, :y], input[i, y+1:])).unsqueeze(0) for i, y in enumerate(target)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        if self.weight != None:
            wc = torch.tensor([self.weight[i] for i in target], device = self.device)
            L = wc*(numerator - torch.log(denominator))
        else:
            L = numerator - torch.log(denominator)
        return -torch.mean(L)

def choose_loss_fnc(config, class_weights=None):

    if config.TRAIN.MARGIN != None:
        if class_weights != None:
            criterion = AngularPenaltySMLoss(config, weight=class_weights)
        else:
            criterion = AngularPenaltySMLoss(config)
    else:
        if class_weights != None:
            criterion = torch.nn.CrossEntropyLoss(weight = class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    return criterion