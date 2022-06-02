from utils import AverageMeter, calculate_metrics, AttrDict, show_cfs_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import ce_loss, consistency_loss, AngularPenaltySMLoss
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
import numpy as np
from ema import ModelEMA
from datetime import datetime,date
import os
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

class SemiSupLearning:
    def __init__(self, model, opt_func="Adam", lr=1e-3, device = 'cpu'):
        self.model = model
        self.opt_func = opt_func
        self.device = device
        self.model.to(self.device);

        ## init
        self.epoch_start = 1
        self.best_valid_perf = None
    def get_dataloader(self, train_dl, valid_dl, test_dl = None):
        self.train_labeled_dl, self.train_unlabeled_dl  = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl



    def get_config(self, config):
        self.config = config
        self.config.TRAIN.EVAL_STEP = len(self.train_unlabeled_dl)

        if self.config.TRAIN.USE_EMA:
            self.ema_model = ModelEMA(model = self.model, decay = self.config.TRAIN.EMA_DECAY, device = self.device)

        self.optimizer = build_optimizer(self.model, opt_func = self.opt_func, lr = self.config.TRAIN.BASE_LR)

        self.lr_scheduler = build_scheduler(config = self.config, optimizer = self.optimizer, n_iter_per_epoch = config.TRAIN.EVAL_STEP)

        if self.config.TRAIN.CLS_WEIGHT:
            self.class_weights = class_weight.compute_class_weight(class_weight  = 'balanced',
                        classes  = np.unique(self.train_labeled_dl.dataset.df[self.config.DATA.TARGET_NAME]).tolist(),
                        y = list(self.train_labeled_dl.dataset.df[self.config.DATA.TARGET_NAME]))

            self.class_weights = torch.tensor(self.class_weights,dtype=torch.float).to(self.device)
        else:
            self.class_weights = None
        
        self.loss_fc = AngularPenaltySMLoss(config, device = self.device)

    def train_one(self, epoch):
        self.model.train()
        labeled_iter = iter(self.train_labeled_dl)
        unlabeled_iter = iter(self.train_unlabeled_dl)
        
        summary_loss = AverageMeter()
        
        tk0 = tqdm(range(self.config.TRAIN.EVAL_STEP), total=self.config.TRAIN.EVAL_STEP)
        
        for batch_idx, _ in enumerate(tk0):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                labeled_iter = iter(self.train_labeled_dl)
                inputs_x, targets_x = labeled_iter.next()
            try:
                (inputs_u_w, inputs_u_s) = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(self.train_unlabeled_dl)
                (inputs_u_w, inputs_u_s) = unlabeled_iter.next()

            bs_lb = inputs_x.shape[0]

            ## split branch
            ## semi-supervised branch
            # inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(self.device, non_blocking=True)
            # print(inputs_x.shape, inputs_u_s.shape)
            inputs_semi_branch = torch.cat((inputs_x, inputs_u_s)).to(self.device, non_blocking=True)
            # input_pseudo_branch =  inputs_u_w
            targets_x = targets_x.to(self.device, non_blocking=True)
            
            # outputs_semi_branch = self.model(inputs_semi_branch)
            # if self.config.TRAIN.USE_EMA:
                # self.ema_model.update(self.model)
            # else:
                # self.model = self.model.to('cpu')
                # output_pseudo_branch = self.model(inputs_u_w.to(self.device))
            if self.config.MODEL.NAME == 'conformer':
                ## out_conv and out_trans
                out_conv, out_trans = self.model(inputs_semi_branch)

                outputs_x = out_trans[:bs_lb]
                # outputs_u_w = out_conv[bs_lb:].chunk(2)[0]
                # outputs_u_s_conv = out_conv[bs_lb:].chunk(2)[1]
                # outputs_u_s_trans = out_trans[bs_lb:].chunk(2)[1]
                outputs_u_s_conv = out_conv[bs_lb:]
                # outputs_u_s_trans = out_trans[bs_lb:]

                # outputs_u_w = output_pseudo_branch[0]
                if self.config.TRAIN.USE_EMA:
                    outputs_u_w = self.ema_model.ema(inputs_u_w.to(self.device))[0]
                else:
                    outputs_u_w = self.model(inputs_u_w.to(self.device))[0]

                del inputs_semi_branch
                # del outputs_semi_branch
                # del output_pseudo_branch

                lx = ce_loss(outputs_x, targets_x, class_weights = self.class_weights, reduction = 'mean')
                lu = consistency_loss(outputs_u_w, outputs_u_s_conv, T = self.config.TRAIN.T, p_cutoff = self.config.TRAIN.THRES, device = self.device)
                
                # lu_trans, mask_trans = consistency_loss(outputs_u_w, outputs_u_s_trans, T = self.config.TRAIN.T, p_cutoff = self.config.TRAIN.THRES)
                # lu = lu_conv + lu_trans
            else:
                """
                if self.config.MODEL.MARGIN:
                fts = self.model.backbone(images)
                losses = self.loss_fc(fts, targets, self.model.fc)
                else:
                outputs = self.model(images)
                losses = ce_loss(outputs, targets, class_weights = self.class_weights, reduction = 'mean')
                """
                if self.config.MODEL.MARGIN != 'None':
                    fts = self.model.backbone(inputs_semi_branch)
                    fts_x, fts_s = fts[:bs_lb], fts[bs_lb:]
                    lx = self.loss_fc(fts_x, targets_x, self.model.fc, self.class_weights)
                    # outputs = self.model(inputs_semi_branch)
                    # outputs_x = outputs[:bs_lb]
                    # outputs_u_s = outputs[bs_lb:]
                    if self.config.TRAIN.USE_EMA:
                        outputs_u_w = self.ema_model.ema(inputs_u_w.to(self.device))
                    else:
                        outputs_u_w = self.model(inputs_u_w.to(self.device))
                    del fts
                    lu = consistency_loss(outputs_u_w, fts_s, T = self.config.TRAIN.T, p_cutoff = self.config.TRAIN.THRES, device = self.device, loss_fc = self.loss_fc, fc = self.model.fc)

                else:
                    outputs = self.model(inputs_semi_branch)
                    outputs_x = outputs[:bs_lb]
                    outputs_u_s = outputs[bs_lb:]
                    if self.config.TRAIN.USE_EMA:
                        outputs_u_w = self.ema_model.ema(inputs_u_w.to(self.device))
                    else:
                        outputs_u_w = self.model(inputs_u_w.to(self.device))
                    # outputs_u_w, outputs_u_s = outputs[bs_lb:].chunk(2)

                    # del inputs
                    del outputs

                    lx = ce_loss(outputs_x, targets_x, class_weights = self.class_weights, reduction = 'mean')
                    lu = consistency_loss(outputs_u_w, outputs_u_s, T = self.config.TRAIN.T, p_cutoff = self.config.TRAIN.THRES, device = self.device)
            
            losses = lx + self.config.TRAIN.LAMBDA_U * lu

            self.optimizer.zero_grad()

            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step_update(epoch * self.config.TRAIN.EVAL_STEP + batch_idx)

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
                
                outputs = eval_model(images)
                if self.config.MODEL.NAME == 'conformer':
                    ## out_conv and out_trans
                    out_conv, out_trans = outputs
                    del outputs
                    outputs = out_trans
                    # loss_conv = ce_loss(out_conv, targets, reduction='mean')            
                    # loss_trans = ce_loss(out_trans, targets, reduction='mean')            
                    # losses = loss_conv + loss_trans
                
                losses = ce_loss(outputs, targets, reduction='mean')            

                summary_loss.update(losses.item(), self.config.DATA.BATCH_SIZE)
                tk0.set_postfix(loss=summary_loss.avg)
                targets = targets.cpu().numpy()
                outputs = F.softmax(outputs, dim=1)
                outputs = outputs.cpu().numpy()
                list_outputs += list(outputs)
                list_targets += list(targets)
            list_outputs = np.array(list_outputs)
            list_outputs = np.argmax(list_outputs, axis=1)
            list_targets = np.array(list_targets)
            metric = calculate_metrics(list_outputs, list_targets)
            if show_metric:
                print('Metric:')
                print(metric)
            if show_report:
                report = classification_report(list_targets, list_outputs)
                print('Classification Report:')
                print(report)
            if show_cf_matrix:
                show_cfs_matrix(list_targets, list_outputs)
            return summary_loss, metric

    

    # def test_one(self, metric = False, report = False, cm = False):
    #     if self.config.TRAIN.USE_EMA:
    #         eval_model = self.ema_model.ema
    #     else:
    #         eval_model = self.model

    #     eval_model.eval()

    #     list_outputs = []
    #     list_targets = []
    #     with torch.no_grad():
            
    #         for step, (images, targets) in tqdm(enumerate(self.valid_dl), total=len(self.valid_dl)):
    #             images = images.to(device, non_blocking=True)
    #             targets = targets.to(device, non_blocking=True)
                
    #             outputs = eval_model(images)
    #             targets = targets.cpu().numpy()
    #             outputs = F.softmax(outputs, dim=1)
    #             outputs = outputs.cpu().numpy()
    #             list_outputs += list(outputs)
    #             list_targets += list(targets)
    #         list_outputs = np.array(list_outputs)
    #         list_outputs = np.argmax(list_outputs, axis=1)
    #         list_targets = np.array(list_targets)
    #         if metric:
    #             metric = calculate_metrics(list_outputs, list_targets)
    #             print('Metric:')
    #             print(f'\t{0}'.format(metric))
    #         elif type_observer == 'report':
    #             report = classification_report(list_targets, list_outputs)
    #             print('Classification Report:')
    #             print(f'\t{0}'.format(report))
    #         elif type_observer == ''
    #     return list_outputs, list_targets

    def save_checkpoint(self, foldname):
        checkpoint = {}

        if self.config.TRAIN.USE_EMA:
            checkpoint['ema_state_dict'] = self.ema_model.ema.state_dict()

        d = date.today().strftime("%m_%d_%Y") 
        h = datetime.now().strftime("%H_%M_%S").split('_')
        h_offset = int(datetime.now().strftime("%H_%M_%S").split('_')[0])+1
        h[0] = str(h_offset)
        h = '_'.join(h)
        filename = d +'_'+h

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
        self.best_valid_perf = checkpoint['best_valid_perf']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        
    def fit(self):
        for epoch in range(self.epoch_start, self.config.TRAIN.EPOCHS+1):
            self.epoch = epoch
            if self.best_valid_perf:
                print(f'Training epoch: {self.epoch} | Current LR: {self.optimizer.param_groups[0]["lr"]:.6f} | The best loss: {float(self.best_valid_perf):.3f}')
            else:
                print(f'Training epoch: {self.epoch} | Current LR: {self.optimizer.param_groups[0]["lr"]:.6f} | The best loss: inf')

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

class SupLearning:
    def __init__(self, model, opt_func="Adam", lr=1e-3, device = 'cpu'):
        self.model = model
        self.opt_func = opt_func
        self.device = device
        self.model.to(self.device);
        ## init
        self.epoch_start = 1
        self.best_valid_perf = None
    def get_dataloader(self, train_dl, valid_dl, test_dl = None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl

    def get_config(self, config):
        self.config = config

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

    def train_one(self, epoch):
        self.model.train()
        

        summary_loss = AverageMeter()
        
        tk0 = tqdm(self.train_dl, total=len(self.train_dl))
        num_steps = len(self.train_dl)

        for step, (images, targets) in enumerate(tk0):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            if self.config.MODEL.MARGIN != 'None':
                fts = self.model.backbone(images)
                losses = self.loss_fc(fts, targets, self.model.fc, self.class_weights)
            else:
                outputs = self.model(images)
                losses = ce_loss(outputs, targets, class_weights = self.class_weights, reduction = 'mean')
            
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
                    
                    outputs = eval_model(images)
                    losses = ce_loss(outputs, targets, reduction='mean')            

                    summary_loss.update(losses.item(), self.config.DATA.BATCH_SIZE)
                    tk0.set_postfix(loss=summary_loss.avg)
                    targets = targets.cpu().numpy()
                    outputs = F.softmax(outputs, dim=1)
                    outputs = outputs.cpu().numpy()
                    list_outputs += list(outputs)
                    list_targets += list(targets)
                list_outputs = np.array(list_outputs)
                list_outputs = np.argmax(list_outputs, axis=1)
                list_targets = np.array(list_targets)
                metric = calculate_metrics(list_outputs, list_targets)
                if show_metric:
                    print('Metric:')
                    print(metric)
                if show_report:
                    report = classification_report(list_targets, list_outputs)
                    print('Classification Report:')
                    print(report)
                if show_cf_matrix:
                    show_cfs_matrix(list_targets, list_outputs)
                return summary_loss, metric

    


    # def test_one(self):
    #     if self.config.TRAIN.USE_EMA:
    #         eval_model = self.ema_model.ema
    #     else:
    #         eval_model = self.model

    #     eval_model.eval()

    #     list_outputs = []
    #     list_targets = []
    #     with torch.no_grad():
            
    #         for step, (images, targets) in tqdm(enumerate(self.valid_dl), total=len(self.valid_dl)):
    #             images = images.to(self.device, non_blocking=True)
    #             targets = targets.to(self.device, non_blocking=True)
                
    #             outputs = eval_model(images)
    #             targets = targets.cpu().numpy()
    #             outputs = F.softmax(outputs, dim=1)
    #             outputs = outputs.cpu().numpy()
    #             list_outputs += list(outputs)
    #             list_targets += list(targets)
    #     list_outputs = [np.argmax(item) for item in list_outputs]
        
    #     return list_outputs, list_targets

    def save_checkpoint(self, foldname):
        checkpoint = {}

        if self.config.TRAIN.USE_EMA:
            checkpoint['ema_state_dict'] = self.ema_model.ema.state_dict()

        d = date.today().strftime("%m_%d_%Y") 
        h = datetime.now().strftime("%H_%M_%S").split('_')
        h_offset = int(datetime.now().strftime("%H_%M_%S").split('_')[0])+1
        h[0] = str(h_offset)
        h = '_'.join(h)
        filename = d +'_'+h

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