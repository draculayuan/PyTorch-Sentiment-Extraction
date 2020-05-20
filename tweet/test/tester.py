import torch
import numpy as np
import time
import torch.nn.functional as F
from .base_tester import BaseTester
from transformers import BertTokenizer

class Tester(BaseTester):
    def __init__(self, model, mode='baseline'):
        super(Tester, self).__init__(model)
        self.mode = mode

    def test(self, data_loader):
        outputs, texts, masks, sel_labels, labels, offsets, rawtexts, rawseltexts, loc_outputs\
                                                            = super().test(data_loader)
        def jaccard(str1, str2): 
            a = set(str1.lower().split()) 
            b = set(str2.lower().split())
            c = a.intersection(b)
            return float(len(c)) / (len(a) + len(b) - len(c))
        jac = 0.0
        preds = []
        
        for (b_outputs, b_texts, b_masks, b_sel_labels,\
             b_labels, b_offsets, b_rawtexts, b_rawseltexts, b_locoutputs)\
        in zip(outputs, texts, masks, sel_labels, labels, offsets, rawtexts, rawseltexts,\
          loc_outputs):
            for idx in range(b_outputs.size(0)):
                if len(b_locoutputs) > 0:
                    start = torch.argmax(b_locoutputs[idx][:, 0], dim=0).item()
                    end = torch.argmax(b_locoutputs[idx][:, 1], dim=0).item()
                    if start > end:
                        start = end
                else:
                    pred_ = torch.argmax(b_outputs[idx], dim=1)
                    pred_[b_masks[idx]==0] = 0
                    start = 1
                    end = sum(b_masks[idx]).item()-2 #to be adjusted for QA mode
                    while start < end and pred_[start].item() != 1:
                        start += 1
                    while start < end and pred_[end].item() != 1:
                        end -= 1
                        
                if b_labels[idx].item() == 0 or len(b_rawtexts[idx].split()) < 2:
                    text_pred = b_rawtexts[idx]
                else:
                    text_pred = ""
                    for i in range(start, end+1):
                        text_pred += b_rawtexts[idx][b_offsets[idx][i][0]:b_offsets[idx][i][1]]
                        if (i+1) < len(b_offsets[idx]) and b_offsets[idx][i][1] < b_offsets[idx][i+1][0]:
                            text_pred += " "
                
                jac += jaccard(text_pred, b_rawseltexts[idx])
                preds.append(text_pred)
        return jac / len(preds), preds
