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
from tweet.data import TweetDataset, tweet_collate_fn
from tweet.utils import AverageMeter, save_checkpoint, load_state, IterLRScheduler, GivenIterationSampler
from tweet.train import Trainer


def main(args):
    # set random seeds
    '''
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #initially true, dont know if ok to turn off
    random.seed(0)
    '''
    # create model
    model = models.create(mode=args.mode)
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    optim = args.optim.lower()
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr,
                                     weight_decay=args.weight_decay)
    # resume from checkpoint
    last_iter = -1
    if args.load_path and args.load_path!='':
        if args.resume_opt:
            last_iter = load_state(args.load_path, model, optimizer=optimizer)
        else:
            load_state(args.load_path, model)

    cudnn.benchmark = True
    # load dataset
    train_dataset = TweetDataset(
        args.train_file,
        args.max_length
    )
    val_dataset = TweetDataset(
        args.test_file,
        args.max_length,
    )

	# learning rate scheduler
    if args.warmup_steps > 0:
        gap = args.warmup_lr - args.base_lr
        warmup_mults = [(args.base_lr + (i+1)*gap/args.warmup_steps) / (args.base_lr + i*gap/args.warmup_steps) for i in range(args.warmup_steps)]
        warmup_steps = list(range(args.warmup_steps))
        args.lr_mults = warmup_mults + args.lr_mults
        args.lr_steps = warmup_steps + args.lr_steps
    lr_scheduler = IterLRScheduler(optimizer, args.lr_steps, args.lr_mults, last_iter=last_iter)
    # trainer
    print('initializing trainer..')
    trainer = Trainer(model, criterion, args.save_path, lr_scheduler, args.job_name,\
                      mode=args.mode, pred_neutral=args.pred_neutral)
    train_sampler = GivenIterationSampler(train_dataset, args.max_iter+32, args.batch_size,
                                                     last_iter=last_iter)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,\
                              pin_memory=False, sampler=train_sampler,\
                              collate_fn=tweet_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=False, collate_fn=tweet_collate_fn)
    start_time = time.time()
    trainer.train(last_iter+1, train_loader, val_loader, optimizer,\
                  args.print_freq, args.val_freq)
    end_time = time.time()
    m, s = divmod(int(end_time - start_time), 60)
    h, m = divmod(m, 60)
    print('Finished training.')
    print('Totally costs {:02d}:{:02d}:{:02d} to train.'.format(h, m, s))


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
