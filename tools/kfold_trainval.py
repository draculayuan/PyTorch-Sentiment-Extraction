import argparse
import yaml
import os
import sys
import time
import math
import random
from easydict import EasyDict
import shutil
import numpy as np
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tweet import models
from tweet.data import KFoldDataset, tweet_collate_fn
from tweet.utils import AverageMeter, save_checkpoint, load_state, IterLRScheduler, GivenIterationSampler
from tweet.train import Trainer
from tweet.test import Tester


def main(args):
    # set random seeds
    '''
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #initially true, dont know if ok to turn off
    random.seed(0)
    '''
    # logical constrains
    assert args.model in ['bert', 'roberta']
    if args.mode == 'embed-cat':
        assert args.model == 'roberta'
        assert args.qa == True
        
    cudnn.benchmark = True
    
    # criterion
    if args.weighted_loss is not None:
        print('\n Using weighted loss \n')
        assert 'sent' not in args.mode # because this loss is also shared with sent head, you cant use the same loss weight for both tasks
        weight = torch.FloatTensor(args.weighted_loss).cuda()
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()
        
    # data
    # load dataset
    train_dataset = KFoldDataset(
        args.train_file,
        args.max_length,
        args.qa,
        args.model,
        args.k
    )
    val_dataset = KFoldDataset(
        args.train_file,
        args.max_length,
        args.qa,
        args.model,
        args.k,
        True
    )
        
    # train-val loop
    for idx in range(args.k):
        # initializing model
        # create model
        model = models.create(model_type=args.model_type, mode=args.mode, model=args.model)
        model.cuda()
        
        # prepare training
        train_dataset.current_fold = idx
        val_dataset.current_fold = idx
        train_sampler = GivenIterationSampler(train_dataset, args.max_iter+32, args.batch_size,
                                                         last_iter=-1)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,\
                                  pin_memory=False, sampler=train_sampler,\
                                  collate_fn=tweet_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=1, pin_memory=False, collate_fn=tweet_collate_fn)
        
        optim = args.optim.lower()
        if optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        elif optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr,
                                         weight_decay=args.weight_decay)

        # learning rate scheduler
        if args.warmup_steps > 0:
            gap = args.warmup_lr - args.base_lr
            warmup_mults = [(args.base_lr + (i+1)*gap/args.warmup_steps) / (args.base_lr + i*gap/args.warmup_steps) for i in range(args.warmup_steps)]
            warmup_steps = list(range(args.warmup_steps))
            args.lr_mults = warmup_mults + args.lr_mults
            args.lr_steps = warmup_steps + args.lr_steps
        lr_scheduler = IterLRScheduler(optimizer, args.lr_steps, args.lr_mults, last_iter=-1)
        # trainer
        print('initializing trainer for fold {}..'.format(idx))
        trainer = Trainer(model, criterion, args.save_path, lr_scheduler, \
                          args.job_name+'_'+'fold'+str(idx),\
                          mode=args.mode, pred_neutral=args.pred_neutral)

        trainer.train(0, train_loader, val_loader, optimizer,\
                      args.print_freq, args.val_freq)
        # -------------------------------------------------------------------------
        model.eval()
        print('Overall validation for fold{}'.format(idx))
        # prepare validation
        tester = Tester(model, args.mode)
        jaccard, _ = tester.test(val_loader)
        print('\n\n test finished, jaccard score: {:.4f}'.format(jaccard))
        model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=\
                                     'Kaggle Tweet Sentiment Description,Single Model based')
    parser.add_argument('--config', default='hey_bro_wheres_your_config')
    parser.add_argument('--job_name',default='hey_bro_you_never_add_name')

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    config['common'].update({'job_name':args.job_name})
    config = EasyDict(config['common'])

    main(config)
