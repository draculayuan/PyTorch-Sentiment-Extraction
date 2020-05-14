from __future__ import absolute_import

import torch
import torchvision
import numpy as np

class BaseTester():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        '''
        for param in self.model.parameters():
            if param.requires_grad == False:
                print(param)
        '''        
    def test(self, data_loader):
        outputs, texts, masks, sel_labels, labels = [], [], [], [], []

        with torch.no_grad():
            for batch_index, (text, mask, sel_label, label, type_id) in enumerate(data_loader):
                text = text.cuda()
                mask = mask.cuda()
                type_id = type_id.cuda()
                output = self.model(text, mask, type_id)[0]
                outputs.append(output)
                texts.append(text)
                masks.append(mask)
                sel_labels.append(sel_label)
                labels.append(label)

        return outputs, texts, masks, sel_labels, labels
