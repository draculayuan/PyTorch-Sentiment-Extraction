import torch
import operator
from torch.autograd import Variable
import torch.nn as nn
from .base_trainer import BaseTrainer

class Trainer(BaseTrainer):

    def __init__(self, model, criterion, save_path, scheduler=None, job_name=None, mode='baseline', pred_neutral=True):
        super(Trainer, self).__init__(model, criterion, save_path, scheduler, job_name)
        self.mode = mode
        self.pred_neutral = pred_neutral
        print('\n\n Trainer Initialized with training mode: {}'.format(self.mode))
        if not self.pred_neutral:
            print('Igoring neutral when predicting sentiment')

    def performance(self, output, text, mask, sel_label, label, offset, rawtext, rawseltext, out_loc=None):
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
            if out_loc is None:
                # NER Style
                temp = torch.argmax(output[idx], dim=1)
                temp[mask[idx]==0] = 0
                start = 1 # skip cls token
                end = sum(mask[idx]).item() - 2 # #to be adjusted for QA mode
                while start <= end and temp[start].item() != 1:
                    start += 1
                while start <= end and temp[end].item() != 1:
                    end -= 1
            else:
                # CLF Style
                start = torch.argmax(out_loc[idx][:,0], dim=0).item()
                end = torch.argmax(out_loc[idx][:,1], dim=0).item()
                if start > end:
                    start = end
                
            if label[idx].item() == 0 or len(rawtext[idx].split()) < 2:
                pred_tok = rawtext[idx]
            else:
                pred_tok = ""
                for i in range(start, end+1):
                    pred_tok += rawtext[idx][offset[idx][i][0]:offset[idx][i][1]]
                    if (i+1) < len(offset[idx]) and offset[idx][i][1] < offset[idx][i+1][0]:
                        pred_tok += " "

            # Step 2, cal jaccard
            jac += jaccard(pred_tok, rawseltext[idx])
            pred_toks.append(pred_tok)
        return jac / output.size(0), pred_toks
