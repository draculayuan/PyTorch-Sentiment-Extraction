from __future__ import absolute_import

import torch
import torchvision
import numpy as np

class BaseInfer():
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def infer(self, data_loader):
        preds, masks, ori_ids, sents, df_ids = [], [], [], [], []

        with torch.no_grad():
            for batch_index, (text, mask, label, ori_tok, df_id) in enumerate(data_loader):
                sents.append(label)
                text = text.cuda()
                mask = mask.cuda()
                pred = self.model(text, mask)[0] #only pred sel text needed
                preds.append(pred)
                masks.append(mask)
                ori_ids.append(text)
                df_ids.append(df_id)

        return preds, masks, ori_ids, sents, df_ids
