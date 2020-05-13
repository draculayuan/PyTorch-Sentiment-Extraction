import torch
import numpy as np
import time
import torch.nn.functional as F
from .base_infer import BaseInfer
from transformers import BertTokenizer

class Infer(BaseInfer):
    def __init__(self, model):
        super(Infer, self).__init__(model)
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)

    def infer(self, data_loader):
        preds, masks, ori_ids, sents, df_ids = super().infer(data_loader)

        text_pred = []
        flat_id = []
        for (b_preds, b_masks, b_oriids, b_sents, b_dfids) in \
                                    zip(preds, masks, ori_ids, sents, df_ids):
            for idx in range(b_preds.size(0)):
                pred_ = torch.argmax(b_preds[idx], dim=1)
                pred_[b_masks[idx]==0] = 0
                start = 1 # start from 1, as first is cls token
                end = sum(b_masks[idx]).item()-2 # this should give the location of the token before sep
                while start < end and pred_[start].item() != 1:
                    start += 1
                while start < end and pred_[end].item() != 1:
                    end -= 1

                if b_sents[idx].item() == 0 or sum(b_masks[idx]).item() <= 5:
                    text_pred_ = b_oriids[idx][1:sum(b_masks[idx]).item()-1]
                else:
                    text_pred_ = b_oriids[idx][start:end+1]
                
                text_pred_ = self.tokenizer.decode(text_pred_)
                text_pred.append(text_pred_)
                flat_id.append(b_dfids[idx])
        
        return text_pred, flat_id
