from __future__ import division
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from . import BaseDataset
import tokenizers
#from transformers import BertTokenizer

class InferDataset(BaseDataset):
    """
      Dataset for Tweet Sentiment Extraction
    """
    def __init__(self, data_path,
                 max_length=64,
                 qa=False,
                 model='bert'
                ):
        super(InferDataset, self).__init__()
        assert model in ['bert', 'roberta']
        self.data_path = data_path
        self.id_list = list()
        self.text_list = list()
        self.label = list()
        self.max_length = max_length
        self.model = model
        self.label_map = {
                    'neutral':0,
                    'negative': 1,
                    'positive': 2
        }
        if self.model == 'bert':
            self.sent_id = {
                        0: 8699,
                        1: 4997,
                        2: 3893
            }
            self.tokenizer = tokenizers.BertWordPieceTokenizer(
                '/home/liu/DL_workstation/tweet-sent/tweet-pytorch/tools/vocab.txt',
                lowercase=True
            )
        elif self.model == 'roberta':
            self.sent_id = {
                        2: 1313,
                        1: 2430,
                        0: 7974
            }
            self.tokenizer = tokenizers.ByteLevelBPETokenizer(
                vocab_file='/home/liu/DL_workstation/tweet-sent/tweet-pytorch/tools/vocab.json', 
                merges_file='/home/liu/DL_workstation/tweet-sent/tweet-pytorch/tools/merges.txt', 
                lowercase=True,
                add_prefix_space=True
            )
        self.qa = qa
        if self.qa:
            print('\n\n Inferring with Question Answering Mode...')
        self.parse(self.data_path)
        self.invalid_cnt = 0

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

    def tokenize_and_getSelLabel(self, tweet, sentiment):
        tok_tweet = self.tokenizer.encode(tweet)
        
        if self.model == 'bert':
            input_ids_orig = tok_tweet.ids[1:-1]
            tweet_offsets = tok_tweet.offsets[1:-1]
            input_ids = [101] + input_ids_orig + [102]
            token_type_ids = [0] * (len(input_ids_orig) + 2)
            mask = [1] * len(token_type_ids)
            tweet_offsets = [(0,0)] + tweet_offsets + [(0,0)]

            if self.qa:
                input_ids += [self.sent_id[sentiment]] + [102]
                token_type_ids += [1] * 2
                mask += [1] * 2
                tweet_offsets += [(0,0)] * 2

            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + ([0] * padding_length)
                mask = mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)
                tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        
        elif self.model == 'roberta':
            input_ids_orig = tok_tweet.ids
            tweet_offsets = tok_tweet.offsets
            input_ids = [0] + input_ids_orig + [2] 
            token_type_ids = [0] * (len(input_ids_orig) + 2)
            mask = [1] * len(token_type_ids)
            tweet_offsets = [(0, 0)] + tweet_offsets + [(0, 0)]
            
            if self.qa:
                input_ids += [2] + [self.sent_id[sentiment]] + [2]
                token_type_ids += [0] * 3 #roberta does not use it during pretrianing, can finetune
                mask += [1] * 3
                tweet_offsets += [(0,0)] * 3
                
            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + ([1] * padding_length)
                mask = mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)
                tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        
        return {
            'ids': input_ids[:self.max_length],
            'mask': mask[:self.max_length],
            'token_type_ids': token_type_ids[:self.max_length],
            'offsets': tweet_offsets[:self.max_length]
        }


    def __getitem__(self, index):
        df_id, text, label = self.id_list[index], self.text_list[index], self.label[index]

        inputs = self.tokenize_and_getSelLabel(text, label) 
        
        return torch.LongTensor(inputs['ids']), torch.LongTensor(inputs['mask']),\
                torch.LongTensor([label]), torch.LongTensor(inputs['token_type_ids']), \
                inputs['offsets'], text, df_id


    def __len__(self):
        return len(self.text_list)

