import torch.nn.functional as F
import torch
import math
import torch.nn as nn

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, epsilon: float = 0.1, 
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon   = epsilon
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, i, j):
        return (1 - self.epsilon) * i + self.epsilon * j

    def forward(self, predict_tensor, target):
        assert 0 <= self.epsilon < 1

        if self.weight is not None:
            self.weight = self.weight.to(predict_tensor.device)

        num_classes = predict_tensor.size(-1)
        
        log_preds = F.log_softmax(predict_tensor, dim=-1)
        
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        
        negative_log_likelihood_loss = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(negative_log_likelihood_loss, loss / num_classes,)

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, class_weights = None, reduction = 'none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
        if class_weights != None:
            self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss

def ce_loss(logits, targets, class_weights = None, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    focal_loss = FocalLoss(gamma = 1, class_weights= class_weights, reduction= reduction)
    if use_hard_labels:
        # return F.cross_entropy(logits, targets, weight=class_weights, reduction=reduction)
        return focal_loss(logits, targets)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return nll_loss

def consistency_loss(logits_w, logits_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True, device = None, loss_fc = None, fc = None):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()


    if loss_fc and fc:
        pseudo_label = torch.softmax(logits_w, dim=-1)
        pseudo_label = pseudo_label.to(device)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()

        masked_loss = loss_fc(logits_s, max_idx, fc, use_mask = 'True', mask = mask)
        # return masked_loss
        return masked_loss, mask.mean()

    else:

        if name == 'L2':
            assert logits_w.size() == logits_s.size()
            return F.mse_loss(logits_s, logits_w, reduction='mean')
        
        elif name == 'L2_mask':
            pass

        elif name == 'ce':
            pseudo_label = torch.softmax(logits_w, dim=-1)
            pseudo_label = pseudo_label.to(device)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff).float()
            # mask = mask
            if use_hard_labels:
                masked_loss = ce_loss(logits = logits_s, 
                                    targets = max_idx,
                                    class_weights = None,
                                    use_hard_labels = use_hard_labels, reduction='none') * mask
            else:
                pseudo_label = torch.softmax(logits_w/T, dim=-1)
                masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
            return masked_loss.mean(), mask.mean()
            # return  masked_loss.mean()

        else:
            assert Exception('Not Implemented consistency_loss')

class TripletLoss(nn.Module):
    """
    Triplet loss
    """
    def __init__(self, alpha, device):
        super(TripletLoss, self).__init__()
        ## alpha --> bias
        self.alpha = alpha
        self.device = device

    def forward(self, anchor, positive, negative, average_loss=True):
        ## Frobenius norm
        d_p = torch.norm(anchor - positive,dim=1)
        d_n = torch.norm(anchor - negative,dim=1)

        losses = torch.max(d_p - d_n + self.alpha, torch.FloatTensor([0]).to(self.device))
        
        if average_loss:
            return losses.mean(), d_p.mean(), d_n.mean()

        return losses.sum(), d_p.mean(), d_n.mean()
        


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, config, s=None, m=None, eps=1e-7, device = None):
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

        self.loss_type = config.MODEL.MARGIN
        if self.loss_type == 'arcface':
            self.s = 30.0 if not s else s
            self.m = 0.3 if not m else m
        if self.loss_type == 'sphereface':
            self.s = 30.0 if not s else s
            self.m = 1.35 if not m else m
        if self.loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        if self.loss_type == 'acloss':
            self.s = 30.0 if not s else s
            self.m = 0.3 if not m else m
        self.eps = eps
        # self.weight = weight
        # self.bn = nn.BatchNorm1d(config.MODEL.NUM_CLASSES)
        self.device = device

    def forward(self, input, target, weight_fc, cls_weight = None, use_mask = False, mask = None):
        '''
        input shape (N, in_features)
        '''
        assert len(input) == len(target)
        assert torch.min(target) >= 0
        
        # input = self.bn(input)
        input = F.normalize(input, p=2, dim=1)

        for w in weight_fc.parameters():
            w = F.normalize(w, p=2, dim=1)
    
        input = weight_fc(input)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(input.transpose(0, 1)[target]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(input.transpose(0, 1)[target]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(input.transpose(0, 1)[target]), -1.+self.eps, 1-self.eps)))
        if self.loss_type == 'acloss':
            acos = torch.acos(torch.clamp(torch.diagonal(input.transpose(0, 1)[target]), -1.+self.eps, 1-self.eps)) + self.m
            numerator = self.s * g_theta(acos)
        excl = torch.cat([torch.cat((input[i, :y], input[i, y+1:])).unsqueeze(0) for i, y in enumerate(target)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        if cls_weight != None:
            cls_weight = torch.tensor([cls_weight[i] for i in target], device=self.device)
            L = cls_weight*(numerator - torch.log(denominator))
        else:
            L = numerator - torch.log(denominator)
        if use_mask:
            L = L * mask
        return -torch.mean(L)

def g_theta(arccos, k = 0.3):
    sigmoid1 = (1+math.exp(-math.pi/2.0/k))/(1-math.exp(-math.pi/2.0/k))
    sigmoid2 = (1-torch.exp(arccos/k-math.pi/2.0/k))/(1+torch.exp(arccos/k-math.pi/2.0/k))
    cos_t = sigmoid1 * sigmoid2
    return cos_t

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