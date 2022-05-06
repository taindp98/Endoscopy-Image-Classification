from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

import torch.nn as nn
from datetime import datetime,date
import os



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

def calculate_metrics(pred, target):
    pred = np.argmax(pred, axis=1)
#     target = np.argmax(target, axis=1)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro')
           }
    

def train_one(data_loader,model,criterion,optimizer, config, device):
    model.train()
    criterion.train()
    
    summary_loss = AverageMeter()
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for step, (images, targets) in enumerate(tk0):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        if config.MODEL.TYPE == 'attguide':
            o_global, o_local = outputs
            l_global = criterion(o_global, targets)
            l_local = criterion(o_local, targets)
            losses = 0.8*l_global + 0.2*l_local
        else:
            losses = criterion(outputs, targets)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
#         if scheduler is not None:
#             scheduler.step()
        
        summary_loss.update(losses.item(), config.DATA.BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)
        
    return summary_loss

def eval_one(data_loader, model,criterion, config, device):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()
    list_outputs = []
    list_targets = []
    with torch.no_grad():
        
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets) in enumerate(tk0):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images, is_valid=True)
            # if config.MODEL.TYPE == 'attguide':
            #     o_global, o_local = outputs
            #     l_global = criterion(o_global, targets)
            #     l_local = criterion(o_local, targets)
            #     losses = 0.8*l_global + 0.2*l_local
            #     outputs = o_global
            # else:
            losses = criterion(outputs, targets)            
            summary_loss.update(losses.item(), config.DATA.BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)
            targets = targets.cpu().numpy()
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs.cpu().numpy()
            list_outputs += list(outputs)
            list_targets += list(targets)
        metric = calculate_metrics(np.array(list_outputs), np.array(list_targets))
    return summary_loss, metric

def inference(data_loader, model, device):
    model.eval()
    list_outputs = []
    list_targets = []
    with torch.no_grad():
        
        for step, (images, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            targets = targets.cpu().numpy()
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs.cpu().numpy()
            list_outputs += list(outputs)
            list_targets += list(targets)
    return list_outputs, list_targets

def save_checkpoint(model, filepath, epoch):
    """
    checkpoint = {
            'model': best_model,
            'epoch':epoch+1,
            'model_state_dict':best_model.state_dict(),
            'optimizer_state_dict':best_optimizer.state_dict(),
            'scheduler_state_dict':best_scheduler.state_dict()
            }
    """
    d = date.today().strftime("%m_%d_%Y") 
    h = datetime.now().strftime("%H_%M_%S").split('_')
    h_offset = int(datetime.now().strftime("%H_%M_%S").split('_')[0])+1
    h[0] = str(h_offset)
    h = '_'.join(h)
    today_time = d +'_'+h

    checkpoint = {
            'model': model,
            'epoch':epoch,
            'model_state_dict':model.state_dict()
            }
    f = os.path.join(filepath, today_time + '.pth')
    torch.save(checkpoint, f)
    
def load_checkpoint(model, filepath, is_train=False):
    checkpoint = torch.load(filepath, map_location = {'cuda:0':'cpu'})    
    # model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    if is_train:
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model

def show_cfs_matrix(targ, pred):
    C = confusion_matrix(targ, pred)
    cmn = C / C.astype('float').sum(axis=1)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    
def show_batch(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig=plt.figure(figsize=(20, 7))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated