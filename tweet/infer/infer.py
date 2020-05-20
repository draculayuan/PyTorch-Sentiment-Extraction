import torch
import numpy as np
from .base_infer import BaseInfer

class Infer(BaseInfer):
    def __init__(self, model, mode):
        super(Infer, self).__init__(model)
        self.mode = mode

    def infer(self, data_loader):
        preds, masks, sents, offsets, rawtexts, dfids, loc_preds = super().infer(data_loader)

        text_pred = []
        flat_id = []
        for (b_preds, b_masks, b_sents, b_offsets, b_rawtexts, b_dfids, b_locpreds) in \
                                    zip(preds, masks, sents, offsets, rawtexts, dfids, loc_preds):
            for idx in range(b_preds.size(0)):
                if len(b_locpreds) > 0:
                    start = torch.argmax(b_locpreds[idx][:, 0], dim=0).item()
                    end = torch.argmax(b_locpreds[idx][:, 1], dim=0).item()
                    if start > end:
                        start = end
                else:
                    pred_ = torch.argmax(b_preds[idx], dim=1)
                    pred_[b_masks[idx]==0] = 0
                    start = 1 # start from 1, as first is cls token
                    end = sum(b_masks[idx]).item()-2 # this should give the location of the token before sep
                    while start < end and pred_[start].item() != 1:
                        start += 1
                    while start < end and pred_[end].item() != 1:
                        end -= 1

                if b_sents[idx].item() == 0 or len(b_rawtexts[idx].split()) < 2:
                    text_pred_ = b_rawtexts[idx]
                else:
                    text_pred_ = ""
                    for i in range(start, end+1):
                        text_pred_ += b_rawtexts[idx][b_offsets[idx][i][0]:b_offsets[idx][i][1]]
                        if (i+1) < len(b_offsets[idx]) and b_offsets[idx][i][1] < b_offsets[idx][i+1][0]:
                            text_pred_ += " "
                
                text_pred.append(text_pred_)
                flat_id.append(b_dfids[idx])
        
        return text_pred, flat_id
