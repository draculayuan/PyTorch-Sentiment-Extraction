from transformers import BertModel, RobertaModel
import torch.nn as nn
import torch
from .model_helper import sentimented_embedding, CNN_clf

mode_fact = [
             'baseline', 
             'sent-clf', 
             'sent-ori', 
             'sent-mask', 
             'sent-sel', 
             'sent-cycle', 
             'sent-sel',
             'embed-cat', # not exactly concatenating, adding actually
             'loc_loss', # baseline with addition start end location loss
             'loc-pure'
            ]
model_fact = {
        'bert': BertModel,
        'roberta': RobertaModel
}

loc_head = {
        'nn': nn.Linear,
        'cnn': CNN_clf
}
class Model(nn.Module):
    def __init__(
        self, 
        model_type,
        output_attentions,
        dropout,
        mode,
        model,
        head,
    ):
        super(Model, self).__init__()
        self.bert = model_fact[model].from_pretrained(
                        model_type,
                        output_attentions = output_attentions,
                        output_hidden_states=True
                     )
        self.drop = nn.Dropout(dropout)
        if 'base' in model_type:
            hidden_dim = 768
        elif 'large' in model_type:
            hidden_dim = 1024
        else:
            raise ValueError('Model Type not implemented.')
        # NER Classifier
        self.clf = nn.Linear(hidden_dim*2, 2)
        torch.nn.init.normal_(self.clf.weight, std=0.02)
        if 'loc' in mode:
            print('Using {} for LOC clf'.format(head))
            self.loc_clf = loc_head[head](hidden_dim*2, 2)
            try:
                torch.nn.init.normal_(self.loc_clf.weight, std=0.02)
            except:
                pass
        # mode
        self.mode = mode
        assert self.mode in mode_fact
        if 'sent' in self.mode:
            print('\n Sentiment Booster: {} \n'.format(self.mode))
            if 'clf' not in self.mode:
                self.sent_feat = nn.GRU(hidden_dim*2, hidden_dim, batch_first=True, bidirectional=True)
            self.sent_clf = nn.Linear(hidden_dim*2, 3) # times 2 cuz bidirect
            torch.nn.init.normal_(self.sent_clf.weight, std=0.02)
            
    def forward(self, batch_texts, batch_masks, batch_types, batch_sel_labels=None):
        
        if self.mode == 'embed-cat':
            _, _, feat = sentimented_embedding(self.bert, 
                                              batch_texts,
                                              token_type_ids = batch_types,
                                              attention_mask = batch_masks)
        else:
            _, _, feat = self.bert(batch_texts,
                                     token_type_ids = batch_types,
                                     attention_mask = batch_masks)
        # feat contains initial embedding + hidden feat of all 12 layers
        feat_backup = feat
        feat = torch.cat((feat[-1], feat[-2]), dim=-1)
        # feat shape: b x seq x 1536
        out = self.drop(feat)
        out = self.clf(out)
        if 'loc' in self.mode:
            loc = self.drop(feat)
            loc = self.loc_clf(loc)
            return [out, loc]
        # depreciated
        if self.mode == 'sent-ori':
            _, sent = self.sent_feat(feat)
            sent = torch.cat((sent[0], sent[1]), dim=1)
            sent = self.sent_clf(sent)
            return [out, sent]
        # depreciated
        elif self.mode == 'sent-mask':
            sent = []
            for idx in range(feat.size(0)):
                feat_ = feat[idx] # seq x 1536
                try:
                    end = (batch_masks[idx] == 0).nonzero()[0].item()
                except:
                    end = feat_.size(0)
                _, sent_ = self.sent_feat(feat_[:end, :].unsqueeze(0))
                sent_ = torch.cat((sent_[0], sent_[1]), dim=1) # 1 x 1536
                sent.append(sent_)
            sent = torch.cat(tuple(sent), dim=0)
            sent = self.sent_clf(sent)
            return [out, sent]
        # depreciated
        elif self.mode == 'sent-clf':
            sent = torch.cat((feat_backup[-1], feat_backup[-2]), dim=-1)[:, 0, :]
            sent = self.sent_clf(sent)
            return [out, sent]
        # depreciated
        elif self.mode == 'sent-sel':
            sent = []
            for idx in range(feat.size(0)):
                feat_ = feat[idx][batch_sel_labels[idx]==1, :] # seq x 1536
                _, sent_ = self.sent_feat(feat_.unsqueeze(0))
                sent_ = torch.cat((sent_[0], sent_[1]), dim=1) # 1 x 1536
                sent.append(sent_)
            sent = torch.cat(tuple(sent), dim=0)
            sent = self.sent_clf(sent)
            return [out, sent]
        return [out]

def create(model_type = "bert-base-uncased",
        output_attentions = False,
        dropout = 0.1,
        mode = 'baseline',
        model = 'bert',
        head = 'nn'):
    return Model(model_type, output_attentions, dropout, mode, model, head)
