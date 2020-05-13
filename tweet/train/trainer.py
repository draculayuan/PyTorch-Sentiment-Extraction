import torch
import operator
from torch.autograd import Variable
import torch.nn as nn
from .base_trainer import BaseTrainer
from transformers import BertTokenizer

class Trainer(BaseTrainer):

    def __init__(self, model, criterion, save_path, scheduler=None, job_name=None, mode='baseline', pred_neutral=True):
        super(Trainer, self).__init__(model, criterion, save_path, scheduler, job_name)
        self.mode = mode
        self.pred_neutral = pred_neutral
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        print('\n\n Trainer Initialized with training mode: {}'.format(self.mode))
        if not self.pred_neutral:
            print('Igoring neutral when predicting sentiment')

    def performance(self, output, text, mask, sel_label, label):
        def jaccard(str1, str2): 
            a = set(str1.lower().split()) 
            b = set(str2.lower().split())
            c = a.intersection(b)
            return float(len(c)) / (len(a) + len(b) - len(c))
        # For each record
        jac = 0.0
        pred_toks = []
        for idx in range(output.size(0)):
            # Step 1, extract pred_tok from origin_tok
            temp = torch.argmax(output[idx], dim=1)
            temp[mask[idx]==0] = 0
            start = 1 # skip cls token
            end = sum(mask[idx]).item() - 2 # skip sep token
            while start <= end and temp[start].item() != 1:
                start += 1
            while start <= end and temp[end].item() != 1:
                end -= 1

            if label[idx].item() == 0 or sum(mask[idx]).item() <= 5:
                pred_tok = text[idx][1:sum(mask[idx]).item()-1]
            else:
                pred_tok = text[idx][start:end+1]
            pred_tok = self.tokenizer.decode(pred_tok)
            sel_tok = text[idx][sel_label[idx]==1]
            sel_tok = self.tokenizer.decode(sel_tok)
            # Step 2, cal jaccard
            jac += jaccard(pred_tok, sel_tok)
            pred_toks.append(pred_tok)
        return jac / output.size(0), pred_toks
