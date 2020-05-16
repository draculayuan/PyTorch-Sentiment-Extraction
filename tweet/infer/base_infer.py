from __future__ import absolute_import

import torch
import torchvision
import numpy as np

class BaseInfer():
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def infer(self, data_loader):
        preds, masks, sents, offsets, rawtexts, dfids = [], [], [], [], [], []

        with torch.no_grad():
            for batch_index, (text, mask, label, type_id, offset, rawtext, dfid) in enumerate(data_loader):
                text = text.cuda()
                mask = mask.cuda()
                type_id = type_id.cuda()
                pred = self.model(text, mask, type_id)[0] #only pred sel text needed
                preds.append(pred)
                masks.append(mask)
                sents.append(label)
                offsets.append(offset)
                rawtexts.append(rawtext)
                dfids.append(dfid)

        return preds, masks, sents, offsets, rawtexts, dfids
