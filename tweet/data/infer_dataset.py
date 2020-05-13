from __future__ import division
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from . import BaseDataset
#for bert tokenization
from transformers import BertTokenizer

class InferDataset(BaseDataset):
    """
      Dataset for Tweet Sentiment Extraction
    """
    def __init__(self, data_path,
                 max_length=64
                ):
        
        super(InferDataset, self).__init__()
        self.data_path = data_path
        self.id_list = list()
        self.text_list = list()
        self.label = list()
        self.max_length = max_length
        self.label_map = {
                    'neutral':0,
                    'negative': 1,
                    'positive': 2
        }
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
        #file.dropna(inplace=True)
        for _, row in file.iterrows():
            text_id = row['textID']
            # raw text
            text = row['text']
            # label
            label = row['sentiment']
            label = self.label_map[label]
            self.id_list.append(text_id)
            self.text_list.append(text)
            self.label.append(label)

    def tokenize_and_getSelLabel(self, text):
        encoded_dict = self.tokenizer.encode_plus(
                        text,
                        add_special_tokens = True,
                        max_length = self.max_length,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        )
        t_text = self.tokenizer.tokenize(text)
        return encoded_dict['input_ids'], encoded_dict['attention_mask'], t_text


    def __getitem__(self, index):
        df_id, text, label = self.id_list[index], self.text_list[index], self.label[index]

        text_ids, text_mask, ori_tokens = self.tokenize_and_getSelLabel(text) # this text_id is not the textID in the original csv file.

        return text_ids.squeeze(), text_mask.squeeze(), torch.LongTensor([label]), ori_tokens, df_id

    def __len__(self):
        return len(self.text_list)

