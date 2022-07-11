# from code.loss import FocalLoss
from utils import AverageMeter, calculate_metrics, AttrDict, show_cfs_matrix, show_batch, show_grid
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import ce_loss, consistency_loss, AngularPenaltySMLoss, TripletLoss, LabelSmoothingLoss, FocalLoss
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
import numpy as np
from ema import ModelEMA
from datetime import datetime,date
import os
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from timm.loss import SoftTargetCrossEntropy

class SupLearning:
    def __init__(self, model, opt_func="Adam", lr=1e-3, device = 'cpu'):
        self.model = model
        self.opt_func = opt_func
        self.device = device
        self.model.to(self.device);
        ## init
        self.epoch_start = 1
        self.best_valid_perf = None
    def get_dataloader(self, train_dl, valid_dl, mixup_fn, test_dl = None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.mixup_fn = mixup_fn
    def get_config(self, config):
        self.config = config
        print('Training mode: Supervised Learning')

        if self.config.TRAIN.USE_EMA:
            self.ema_model = ModelEMA(model = self.model, decay = self.config.TRAIN.EMA_DECAY, device = self.device)

        self.optimizer = build_optimizer(self.model, opt_func = self.opt_func, lr = self.config.TRAIN.BASE_LR)
        self.lr_scheduler = build_scheduler(config = self.config, optimizer = self.optimizer, n_iter_per_epoch = len(self.train_dl))
        if self.config.TRAIN.CLS_WEIGHT:
            self.class_weights = class_weight.compute_class_weight(class_weight  = 'balanced',
                        classes  = np.unique(self.train_dl.dataset.df[self.config.DATA.TARGET_NAME]).tolist(),
                        y = list(self.train_dl.dataset.df[self.config.DATA.TARGET_NAME]))

            self.class_weights = torch.tensor(self.class_weights,dtype=torch.float).to(self.device)
        else:
            self.class_weights = None

        self.loss_fc = AngularPenaltySMLoss(config, device = self.device)
        self.loss_triplet = TripletLoss(alpha = 0.7, device = self.device)

        # if self.config.TRAIN.MIXUP > 0.:
        #  # smoothing is handled with mixup label transform
        #     self.criterion = SoftTargetCrossEntropy()
        # elif self.config.TRAIN.LABEL_SMOOTHING > 0.:
        #     self.criterion = LabelSmoothingLoss(epsilon = config.TRAIN.LABEL_SMOOTHING, weight = self.class_weights)
        # else:
        #     # self.criterion = torch.nn.CrossEntropyLoss()
        #     self.criterion = FocalLoss(gamma = 1, class_weights= self.class_weights, reduction= 'mean')
        # print('Loss fnc: ', self.criterion)
    def train_one(self, epoch):
        self.model.train()
        

        summary_loss = AverageMeter()
        
        tk0 = tqdm(self.train_dl, total=len(self.train_dl))
        num_steps = len(self.train_dl)

        for step, (images, targets) in enumerate(tk0):
            if self.config.MODEL.IS_TRIPLET:
                anchors, poss, negs = images
                targets = targets[0].to(self.device, non_blocking=True)
                imgs = torch.cat([anchors, poss, negs], dim=0).to(self.device, non_blocking=True)

                bs = imgs.size(0)//3

                logits, features = self.model(imgs)
                anchor_logits = logits[:bs]
                anchor_fts = features[:bs]

                pos_fts, neg_fts = torch.split(features[bs:], bs)

                triplet_losses, ap, an = self.loss_triplet(anchor_fts,pos_fts,neg_fts, average_loss=True)
                ce_losses = ce_loss(anchor_logits, targets, class_weights = self.class_weights, reduction = 'mean')
                # ce_losses = self.criterion(anchor_logits, targets)
                losses = ce_losses + 2*triplet_losses
            else:
                if self.mixup_fn is not None:
                    images, targets = self.mixup_fn(images, targets)
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if self.config.MODEL.MARGIN != 'None':
                    fts = self.model.backbone(images)
                    losses = self.loss_fc(fts, targets, self.model.fc, self.class_weights)
                else:
                    outputs = self.model(images)
                    losses = ce_loss(outputs, targets)
            
            self.optimizer.zero_grad()

            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step_update(epoch * num_steps + step)

            if self.config.TRAIN.USE_EMA:
                self.ema_model.update(self.model)
            self.model.zero_grad()

            summary_loss.update(losses.item(), self.config.DATA.BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)
            
        return summary_loss

    def evaluate_one(self, show_metric = False, show_report = False, show_cf_matrix = False):

        if self.config.TRAIN.USE_EMA:
            eval_model = self.ema_model.ema
        else:
            eval_model = self.model

        eval_model.eval()
        
        summary_loss = AverageMeter()
        list_outputs = []
        list_targets = []

        with torch.no_grad():
            
            tk0 = tqdm(self.valid_dl, total=len(self.valid_dl))
            for step, (images, targets) in enumerate(tk0):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                
                if self.config.MODEL.IS_TRIPLET:
                    outputs, _ = eval_model(images)
                else:
                    outputs = eval_model(images)
                losses = ce_loss(outputs, targets, reduction='mean')            
                summary_loss.update(losses.item(), self.config.DATA.BATCH_SIZE)
                tk0.set_postfix(loss=summary_loss.avg)
                targets = targets.cpu().numpy()
                outputs = F.softmax(outputs, dim=1)
                outputs = outputs.cpu().numpy()
                list_outputs += list(outputs)
                list_targets += list(targets)
            arr_outputs = np.array(list_outputs)
            arr_outputs = np.argmax(arr_outputs, axis=1)
            arr_targets = np.array(list_targets)
            metric = calculate_metrics(arr_outputs, arr_targets, self.config)
            if show_metric:
                print('Metric:')
                print(metric)
            if show_report:
                report = classification_report(arr_targets, arr_outputs)
                print('Classification Report:')
                print(report)
            if show_cf_matrix:
                show_cfs_matrix(arr_targets, arr_outputs)
            return summary_loss, metric

    def test_one(self, show_metric = False, show_report = False, show_cf_matrix = False):

        if self.config.TRAIN.USE_EMA:
            eval_model = self.ema_model.ema
        else:
            eval_model = self.model

        eval_model.eval()
        
        summary_loss = AverageMeter()
        list_outputs = []
        list_targets = []

        incorrect_examples = []
        incorrect_labels = []
        incorrect_pred = []

        with torch.no_grad():
            
            tk0 = tqdm(self.valid_dl, total=len(self.valid_dl))
            for step, (images, targets) in enumerate(tk0):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if self.config.MODEL.NAME == 'conformer':
                    ## out_conv and out_trans
                    out_conv, out_trans = eval_model(images)
                    outputs = out_trans + out_conv
                elif self.config.MODEL.IS_TRIPLET:
                    outputs, _ = eval_model(images)
                else:
                    outputs = eval_model(images)
                losses = ce_loss(outputs, targets, reduction='mean')            
                summary_loss.update(losses.item(), self.config.DATA.BATCH_SIZE)
                tk0.set_postfix(loss=summary_loss.avg)
                targets = targets.cpu().numpy()
                outputs = F.softmax(outputs, dim=1)
                outputs = outputs.cpu().numpy()
                list_outputs += list(outputs)
                list_targets += list(targets)
            arr_outputs = np.array(list_outputs)
            arr_outputs = np.argmax(arr_outputs, axis=1)
            arr_targets = np.array(list_targets)
            idxs_mask = ((torch.tensor(arr_outputs) == torch.tensor(arr_targets).view_as(torch.tensor(arr_outputs)))==False).view(-1)
            return idxs_mask

    def inference(self, dl_test):

        if self.config.TRAIN.USE_EMA:
            eval_model = self.ema_model.ema
        else:
            eval_model = self.model

        eval_model.eval()
        
        list_outputs = []
        list_index = []
        with torch.no_grad():
            
            tk0 = tqdm(dl_test, total=len(dl_test))
            for step, (images, index) in enumerate(tk0):
                images = images.to(self.device, non_blocking=True)
                
                if self.config.MODEL.NAME == 'conformer':
                    ## out_conv and out_trans
                    out_conv, out_trans = eval_model(images)
                    outputs = out_trans + out_conv
                elif self.config.MODEL.IS_TRIPLET:
                    outputs, _ = eval_model(images)
                else:
                    outputs = eval_model(images)
                
                outputs = F.softmax(outputs, dim=1)
                outputs = outputs.cpu().numpy()
                list_outputs += list(outputs)
                list_index += np.array(index).tolist()
            list_outputs = np.array(list_outputs)
            list_max_value = np.max(list_outputs, axis=1)
            list_max_cond = np.where(list_max_value > self.config.TRAIN.THRES, 1, 0)
            list_max_idx = np.argmax(list_outputs, axis=1)
            list_preds = list(list_max_idx * list_max_cond)
        dict_preds = dict(zip(list_index, list_preds))
        return dict_preds


    def save_checkpoint(self, foldname):
        checkpoint = {}

        if self.config.TRAIN.USE_EMA:
            checkpoint['ema_state_dict'] = self.ema_model.ema.state_dict()

        d = date.today().strftime("%m_%d_%Y") 
        h = datetime.now().strftime("%H_%M_%S").split('_')
        h_offset = int(datetime.now().strftime("%H_%M_%S").split('_')[0])+2
        h[0] = str(h_offset)
        h = '_'.join(h)
        filename =  d +'_' + h + '_epoch_' + str(self.epoch)

        checkpoint['epoch'] = self.epoch
        checkpoint['best_valid_perf'] = self.best_valid_perf
        checkpoint['model_state_dict'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['scheduler'] = self.lr_scheduler.state_dict()

        f = os.path.join(foldname, filename + '.pth')
        torch.save(checkpoint, f)
        print('Saved checkpoint')

    def load_checkpoint(self, checkpoint_dir, is_train=False):
        checkpoint = torch.load(checkpoint_dir, map_location = {'cuda:0':'cpu'})    
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if is_train:
            for parameter in self.model.parameters():
                parameter.requires_grad = True
        else:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        if self.config.TRAIN.USE_EMA:
            self.ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
            if is_train:
                for parameter in self.ema_model.ema.parameters():
                    parameter.requires_grad = True
            else:
                for parameter in self.ema_model.ema.parameters():
                    parameter.requires_grad = False
        self.epoch_start = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler'])

    def fit(self):
        
        for epoch in range(self.epoch_start, self.config.TRAIN.EPOCHS+1):
            self.epoch = epoch
            print(f'Training epoch: {self.epoch} | Current LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            train_loss = self.train_one(self.epoch)
            print(f'\tTrain Loss: {train_loss.avg:.3f}')
            if (epoch)% self.config.TRAIN.FREQ_EVAL == 0:
                valid_loss, valid_metric = self.evaluate_one()
                if self.best_valid_perf:
                    if self.best_valid_perf > valid_loss.avg:
                        self.best_valid_perf = valid_loss.avg
                        self.save_checkpoint(self.config.TRAIN.SAVE_CP)
                else:
                    self.best_valid_perf = valid_loss.avg
                    self.save_checkpoint(self.config.TRAIN.SAVE_CP)
                print(f'\tValid Loss: {valid_loss.avg:.3f}')
                print(f'\tMetric: {valid_metric}')