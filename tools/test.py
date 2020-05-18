import argparse
import yaml
import os
import os.path as osp
import sys
import time
import shutil
from easydict import EasyDict
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

import tweet
from tweet import models
from tweet.data import TweetDataset, tweet_collate_fn
from tweet.utils import load_state


def main(args):
    # logical constraints
    assert args.model in ['bert', 'roberta']
    if args.mode == 'embed-cat':
        assert args.model == 'roberta'
        assert args.qa == True
    
    torch.backends.cudnn.benchmark = True
    # init model
    model = models.create(model_type=args.model_type, mode=args.mode, model=args.model)
        
    model.cuda()
    if osp.isfile(args.checkpoint):
        load_state(args.checkpoint, model, test=True)
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(args.checkpoint))

    # init tester
    print('initializing tester...')
    tester = tweet.test.Tester(model)

    test_dataset = TweetDataset(
        args.test_file,
        args.max_length,
        args.qa,
        args.model
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=False, collate_fn=tweet_collate_fn)

    # test
    jaccard, preds = tester.test(test_loader)
    # print results
    print('\n\n test finished, jaccard score: {:.4f}'.format(jaccard))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test")

    # Model
    parser.add_argument('--checkpoint', type=str, default='', metavar='PATH',
                        help="Directory to the checkpoint to be test.")
    # Data
    parser.add_argument('--config', default='cfgs/test/config_res50_test.yaml')

    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)

    main(args)
