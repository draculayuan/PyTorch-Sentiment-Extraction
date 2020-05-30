from __future__ import absolute_import

import torch
import torchvision
import torch.distributed as dist
import numpy as np
import sys
from ..utils import dist_init, DistModule, reduce_gradients, AverageMeter, save_checkpoint, IterLRScheduler, get_time
from ..loss import loc_loss
from tensorboardX import SummaryWriter
import time
import random

class BaseTrainer():
    def __init__(self, model, criterion, save_path, scheduler=None, job_name=None):
        self.model = model
        self.save_path = save_path
        self.logger = SummaryWriter(save_path)
        self.scheduler = scheduler
        self.fn_save = save_checkpoint
        self.job_name = job_name
        self.criterion = criterion
        self.best_perf = 0

    def train(self, start_iter, train_loader, val_loader, optimizer, print_freq=100, val_freq=500):
        train_losses = AverageMeter(10)
        train_acc = AverageMeter(10)
        
        for batch_index, (text, mask, sel_label, label, type_id, offsets, rawtext, rawseltext)\
                                                                        in enumerate(train_loader):
            curr_step = start_iter+batch_index
            self.scheduler.step(curr_step) # for model optimizer
            lr = optimizer.param_groups[0]['lr']
            
            text = text.cuda()
            mask = mask.cuda()
            sel_label = sel_label.cuda()
            type_id = type_id.cuda()
            label = label.cuda(async=True)
            
            loss = 0
            
            out = self.model(text, mask, type_id, sel_label)
            if self.mode in ['baseline', 'embed-cat']:
                out = out[0]
            elif 'loc' in self.mode:
                out, out_loc = out
                loss += loc_loss(out_loc, sel_label)
            else:
                out, out_sent = out
            #TODO
            # out shape bs x seq x 2, sel_label shape bs x seq
            '''
            loss_sent = 0
            if 'sent' in self.mode:
                if self.pred_neutral:
                    loss_sent += self.criterion(out_sent, label.squeeze())
                else:
                    out_sent_ = out_sent[label.squeeze()!= 0,:]
                    label_ = label[label.squeeze()!=0].squeeze()
                    if out_sent_.size(0) == 0:
                        pass
                    else:
                        loss_sent += self.criterion(out_sent_, label_)
            '''
            loss_sel = 0
            if 'loc' in self.mode:
                if 'pure' not in self.mode:
                    for dim in range(out.size(1)): # iterate over seq length
                        loss_sel += self.criterion(out[:, dim, :], sel_label[:, dim])
                acc_jac, _ = self.performance(out, text, mask, sel_label, label, offsets, rawtext, rawseltext, out_loc=out_loc)
            else:
                for dim in range(out.size(1)): # iterate over seq length
                    loss_sel += self.criterion(out[:, dim, :], sel_label[:, dim])
                acc_jac, _ = self.performance(out, text, mask, sel_label, label, offsets, rawtext, rawseltext)
            
            loss += loss_sel # + loss_sent
            reduced_loss = loss.data.clone()
            reduced_acc = acc_jac

            train_losses.update(reduced_loss.item())
            train_acc.update(reduced_acc)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print and log
            if curr_step > 1 and curr_step % print_freq == 0:
                print('[Train] Time:{} Iter:{} Loss:{:.4f} Acc:{:.4f}'.format(
                    get_time(), curr_step, train_losses.avg, train_acc.avg))
                info = {
                    'lr': lr,
                    'train_loss': train_losses.avg,
                    'train_acc': train_acc.avg
                }
                for key, value in info.items():
                    self.logger.add_scalar(key, value, curr_step + 1)
            # save model
            if curr_step > 1 and curr_step % 1000==0:
                self.fn_save({
                    'step':curr_step,
                    'state_dict':self.model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                 }, False, self.save_path + '/' + str(curr_step) + '_'+self.job_name+'.ckpt')
            # validation
            if curr_step > 0 and curr_step % val_freq == 0:
                self.model.eval()
                self.validation(val_loader, curr_step, optimizer)
                self.model.train()

    def validation(self, val_loader, curr_step, optimizer):
        val_losses = AverageMeter(0)
        val_acc = AverageMeter(0)
        self.model.eval()
        for batch_index, (text, mask, sel_label, label, type_id, offsets, rawtext, rawseltext)\
                                                    in enumerate(val_loader):
            text = text.cuda()
            mask = mask.cuda()
            sel_label = sel_label.cuda()
            type_id = type_id.cuda()
            label = label.cuda(async=True)
            with torch.no_grad():
                loss_sel = 0
                if 'loc' in self.mode: # CLF dominants
                    out, out_loc = self.model(text, mask, type_id, sel_label)
                    acc_sel, _ = self.performance(out, text, mask, sel_label, label, offsets, rawtext, rawseltext, out_loc=out_loc)
                    loss_sel += loc_loss(out_loc, sel_label)
                else:
                    out = self.model(text, mask, type_id, sel_label)[0]
                    acc_sel, _ = self.performance(out, text, mask, sel_label, label, offsets, rawtext, rawseltext)
                    for dim in range(out.size(1)): # iterate over seq length
                        loss_sel += self.criterion(out[:, dim, :], sel_label[:, dim])
                reduced_loss = loss_sel.data.clone()
                reduced_acc = acc_sel
                      
                val_losses.update(reduced_loss.item())
                val_acc.update(reduced_acc)

        print('[Validation] Time:{} Loss:{:.4f} Acc:{:.4f}'.format(
            get_time(), val_losses.avg, val_acc.avg))
        self.model.train()
        # update best model
        if val_acc.avg > self.best_perf:
            print('Updating best performing model...')
            self.best_perf = val_acc.avg
            self.fn_save({
                'step':curr_step,
                'state_dict':self.model.state_dict(),
                'optimizer':optimizer.state_dict(),
             }, False, self.save_path + '/' + 'best' + '_'+self.job_name+'.ckpt')
    def performance(self, outputs, labels):
        """ To be override """
        raise NotImplementedError

