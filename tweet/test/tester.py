import torch
import numpy as np
import time
import torch.nn.functional as F
from .base_tester import BaseTester
from transformers import BertTokenizer

class Tester(BaseTester):
    def __init__(self, model):
        super(Tester, self).__init__(model)
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)

    def test(self, data_loader):
        outputs, texts, masks, sel_labels, labels = super().test(data_loader)
        def jaccard(str1, str2): 
            a = set(str1.lower().split()) 
            b = set(str2.lower().split())
            c = a.intersection(b)
            return float(len(c)) / (len(a) + len(b) - len(c))
        jac = 0.0
        preds = []
        
        for (b_outputs, b_texts, b_masks, b_sel_labels, b_labels) in \
                                    zip(outputs, texts, masks, sel_labels, labels):
            for idx in range(b_outputs.size(0)):
                pred_ = torch.argmax(b_outputs[idx], dim=1)
                pred_[b_masks[idx]==0] = 0
                start = 1
                end = sum(b_masks[idx]).item()-2
                while start < end and pred_[start].item() != 1:
                    start += 1
                while start < end and pred_[end].item() != 1:
                    end -= 1
                if b_labels[idx].item() == 0 or sum(b_masks[idx]).item() <= 5:
                    text_pred = b_texts[idx][1:sum(b_masks[idx]).item()-1]
                else:
                    text_pred = b_texts[idx][start:end+1]
                text_pred = self.tokenizer.decode(text_pred)
                text_sel = b_texts[idx][b_sel_labels[idx]==1]
                text_sel = self.tokenizer.decode(text_sel)
                
                jac += jaccard(text_pred, text_sel)
                preds.append(text_pred)
        return jac / len(preds), preds
