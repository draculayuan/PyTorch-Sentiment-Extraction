from transformers import BertModel, RobertaModel
import torch.nn as nn
import torch
from .model_helper import sentimented_embedding, CNN_clf, LSTM_clf, GRU_clf

mode_fact = [
             'baseline', 
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
        'cnn': CNN_clf,
        'lstm': LSTM_clf,
        'gru': GRU_clf
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

        return [out]

def create(model_type = "bert-base-uncased",
        output_attentions = False,
        dropout = 0.1,
        mode = 'baseline',
        model = 'bert',
        head = 'nn'):
    return Model(model_type, output_attentions, dropout, mode, model, head)
