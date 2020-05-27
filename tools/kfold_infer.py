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
    # logical constrains
    assert args.model in ['bert', 'roberta']
    if args.mode == 'embed-cat':
        assert args.model == 'roberta'
        assert args.qa == True
        
    torch.backends.cudnn.benchmark = True
    
    # init inference
    infer_dataset = InferDataset(
        args.test_file,
        args.max_length,
        args.qa,
        args.model
    )
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=infer_collate_fn)

    raw_output = []
    # init model
    for fold in range(args.k):
        try:
            del inferer
            del model
        except:
            pass
        model = models.create(model_type=args.model_type, mode=args.mode, model=args.model, head=args.head)
        model.cuda()
        try:
            load_state(args.save_path+'/'+str(args.iter)\
                       +'_'+args.job_name+'_fold'+str(fold)\
                       +'.ckpt.pth.tar',\
                       model, test=True)
        except:
            raise ValueError("==> No model found at: {}".\
                             format(args.save_path+'/'+str(args.iter)\
                                    +'_'+args.job_name+'_fold'+str(fold)\
                                    +'.ckpt.pth.tar'))
        #infer
        inferer = tweet.infer.Infer(model, args.mode)
        raw_output.append(inferer.get_raw(infer_loader))
        
    # ensemble
    preds_start = []
    preds_end = []
    for f_output in raw_output:
        f_preds_start = torch.Tensor([]).cuda()
        f_preds_end = torch.Tensor([]).cuda()
        for f_out_ in f_output:
            # -1 is loc pred, 0 is pred
            f_preds_start = torch.cat((f_preds_start, f_out_[-1][:,0].unsqueeze(0)), dim=0)
            f_preds_end = torch.cat((f_preds_end, f_out_[-1][:,1].unsqueeze(0)), dim=0)
        preds_start.append(f_preds_start)
        preds_end.append(f_preds_end)

    preds_start = sum(preds_start) / args.k
    preds_end = sum(preds_end) / args.k
    # size num_samples x max_len
    preds = []
    ids = []
    for idx in range(preds_start.size(0)):
        start = torch.argmax(preds_start[idx], dim=0).item()
        end = torch.argmax(preds_end[idx], dim=0).item()
        if start > end:
            start = end
        if raw_output[0][idx][2] == 0 or len(raw_output[0][idx][4].split()) < 2:
            pred_ = raw_output[0][idx][4]
        else:
            pred_ = ""
            for i in range(start, end+1):
                offset_start = raw_output[0][idx][3][i][0]
                offset_end = raw_output[0][idx][3][i][1]
                pred_ += raw_output[0][idx][4][offset_start:offset_end]
                if (i+1) < len(raw_output[0][idx][3]) and \
                offset_end < raw_output[0][idx][3][i+1][0]:
                    pred_ += " "
        preds.append(pred_)
        ids.append(raw_output[0][idx][-2])
    
    result = pd.DataFrame({
        'selected_text': preds,
        'textID': ids
    })
    result = result[['textID', 'selected_text']]
    result.to_csv(args.out_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test")

    # Data
    parser.add_argument('--iter', default=None)
    parser.add_argument('--job_name', default='training')
    parser.add_argument('--config', default='config/i_dont_know')
    parser.add_argument('--out_path', default='submission.csv')

    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)

    main(args)
