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
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

import tweet
from tweet import models
from tweet.data import InferDataset, infer_collate_fn
from tweet.utils import load_state
            
def main(args):
    assert args.model in ['bert', 'roberta']
    torch.backends.cudnn.benchmark = True

    # init model
    model = models.create(model_type=args.model_type, mode=args.mode, model=args.model)
    model.cuda()
    if osp.isfile(args.checkpoint):
        load_state(args.checkpoint, model, test=True)
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(args.checkpoint))

    # init inference
    print('initializing inferencing...')
    inferer = tweet.infer.Infer(model)

    infer_dataset = InferDataset(
        args.test_file,
        args.max_length,
        args.qa,
        args.model
    )
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=False, collate_fn=infer_collate_fn)

    #infer
    preds, ids = inferer.infer(infer_loader)
    result = pd.DataFrame({
        'selected_text': preds,
        'textID': ids
    })
    result = result[['textID', 'selected_text']]
    result.to_csv(args.out_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test")

    # Model
    parser.add_argument('--checkpoint', type=str, default='', metavar='PATH',
                        help="Directory to the checkpoint to be test.")
    # Data
    parser.add_argument('--config', default='config/i_dont_know')
    
    parser.add_argument('--out_path', default='submission.csv')

    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)

    main(args)
