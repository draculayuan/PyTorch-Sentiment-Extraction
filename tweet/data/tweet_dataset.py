from __future__ import division
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from . import BaseDataset
#for bert tokenization
from transformers import BertTokenizer

class TweetDataset(BaseDataset):
    """
      Dataset for Tweet Sentiment Extraction
    """
    def __init__(self, data_path,
                 max_length=64,
                 qa=False
                ):
        
        super(TweetDataset, self).__init__()
        self.data_path = data_path
        self.text_list = list()
        self.sel_text_list = list()
        self.label = list()
        self.max_length = max_length
        self.label_map = {
                    'neutral':0,
                    'negative': 1,
                    'positive': 2
        }
        self.sent_id = {
                    0: 8699,
                    1: 4997,
                    2: 3893
        }
        self.qa = qa # question answering mode
        if self.qa:
            print('\n\n Running with Question Answering Mode...')
        self.parse(self.data_path)
        self.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased',
                do_lower_case=True
        )
        self.invalid_cnt = 0
        self.buffer_cache = None

    def parse(self, data_path):
        assert data_path[-3:] == 'csv'
        file = pd.read_csv(data_path)
        file.dropna(inplace=True)
        for _, row in file.iterrows():
            # raw text
            text = row['text']
            sel_text = row['selected_text']
            # label
            label = row['sentiment']
            if self.label_map.get(label) is None:
                continue
            label = self.label_map[label]
            self.text_list.append(text)
            self.sel_text_list.append(sel_text)
            self.label.append(label)

    def tokenize_and_getSelLabel(self, text, sel_text, sent):
        encoded_dict = self.tokenizer.encode_plus(
                        text,
                        add_special_tokens = True,
                        max_length = self.max_length,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        )
        t_text = self.tokenizer.tokenize(text)
        t_sel_text = self.tokenizer.tokenize(sel_text) 
        sel_len = len(t_sel_text)

        for i in range(len(t_text)-sel_len+1):
            if t_text[i:i+sel_len] == t_sel_text:
                start = i
                end = i + sel_len - 1
                break
        sel_label = torch.zeros_like(encoded_dict['input_ids'])
        # offset from tokenizer
        start += 1
        end += 1
        sel_label[:, start:end+1] = 1.0
        type_id = torch.zeros(encoded_dict['input_ids'].size()).long()
        
        # initailize buffer
        if self.buffer_cache is None:
            self.buffer_cache = tuple([encoded_dict['input_ids'], encoded_dict['attention_mask'], sel_label, type_id])
        
        if self.qa:
            insert_loc = encoded_dict['input_ids'].size(1) - 1
            while encoded_dict['input_ids'][0, insert_loc].item() == 0:
                insert_loc -= 1
            insert_loc += 1
            # need at least 2 spaces
            if insert_loc >= encoded_dict['input_ids'].size(1)-1:
                pass
            else:
                encoded_dict['input_ids'][0, insert_loc] = self.sent_id[sent]
                encoded_dict['input_ids'][0, insert_loc + 1] = 102
                encoded_dict['attention_mask'][0, insert_loc:insert_loc+2] = 1
                type_id[0, insert_loc:] = 1
            if self.buffer_cache is None:
                self.buffer_cache = tuple([encoded_dict['input_ids'], encoded_dict['attention_mask'], sel_label, type_id])

        return encoded_dict['input_ids'], encoded_dict['attention_mask'], sel_label, type_id


    def __getitem__(self, index):
        text, sel_text, label = self.text_list[index], self.sel_text_list[index],\
                                                    self.label[index]
        try:
            text_ids, text_mask, sel_text_label, type_id = \
            self.tokenize_and_getSelLabel(text, sel_text, label)
        
        except:
            self.invalid_cnt += 1
            text_ids, text_mask, sel_text_label, type_id = self.buffer_cache
        
        return text_ids.squeeze(), text_mask.squeeze(), sel_text_label.squeeze(), torch.LongTensor([label]), type_id.squeeze()

    def __len__(self):
        return len(self.text_list)

