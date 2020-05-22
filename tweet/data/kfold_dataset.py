from __future__ import division
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from . import BaseDataset
import tokenizers
from sklearn.model_selection import StratifiedKFold
#from transformers import BertTokenizer

class KFoldDataset(BaseDataset):
    """
      Dataset for Tweet Sentiment Extraction
    """
    def __init__(self, data_path,
                 max_length=64,
                 qa=False,
                 model='bert',
                 k=10,
                 test=False
                ):
        super(KFoldDataset, self).__init__()
        assert model in ['bert', 'roberta']
        self.data_path = data_path
        self.text_list = list()
        self.train_text = list()
        self.val_text = list()
        self.sel_text_list = list()
        self.train_sel_text = list()
        self.val_sel_text = list()
        self.label = list()
        self.train_label = list()
        self.val_label = list()
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
        self.qa = qa # question answering mode
        if self.qa:
            print('\n\n Running with Question Answering Mode...')
        self.parse(self.data_path)
        self.invalid_cnt = 0
        # create k folds
        self.stratify(k)
        self.current_fold = None
        self.test = test
        
    def stratify(self, k):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42) #i just like the number of 42
        generator = skf.split(self.text_list, self.label)
        for _, (t, v) in enumerate(generator):
            self.train_text.append([self.text_list[e] for e in t])
            self.train_sel_text.append([self.sel_text_list[e] for e in t])
            self.train_label.append([self.label[e] for e in t])
            self.val_text.append([self.text_list[e] for e in v])
            self.val_sel_text.append([self.sel_text_list[e] for e in v])
            self.val_label.append([self.label[e] for e in v])
        
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

    def tokenize_and_getSelLabel(self, tweet, selected_text, sentiment):
        len_st = len(selected_text)
        idx0 = None
        idx1 = None
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind: ind+len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        tok_tweet = self.tokenizer.encode(tweet)
        if self.model == 'bert':
            input_ids_orig = tok_tweet.ids[1:-1]
            tweet_offsets = tok_tweet.offsets[1:-1]
        elif self.model == 'roberta':
            input_ids_orig = tok_tweet.ids
            tweet_offsets = tok_tweet.offsets

        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        if self.model == 'bert':
            input_ids = [101] + input_ids_orig + [102]
            token_type_ids = [0] * (len(input_ids_orig) + 2)
            mask = [1] * len(token_type_ids)
            tweet_offsets = [(0,0)] + tweet_offsets + [(0,0)]
            targets_start += 1
            targets_end += 1
            sel_label = [0] * len(input_ids)
            for idx in range(targets_start, targets_end+1):
                sel_label[idx] = 1

            if self.qa:
                input_ids += [self.sent_id[sentiment]] + [102]
                token_type_ids += [1] * 2
                mask += [1] * 2
                tweet_offsets += [(0,0)] * 2
                sel_label += [0] * 2
                
            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + ([0] * padding_length)
                mask = mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)
                tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
                sel_label = sel_label + [0] * padding_length

        elif self.model == 'roberta':
            input_ids = [0] + input_ids_orig + [2] 
            token_type_ids = [0] * (len(input_ids_orig) + 2)
            mask = [1] * len(token_type_ids)
            tweet_offsets = [(0, 0)] + tweet_offsets + [(0, 0)]
            targets_start += 1
            targets_end += 1
            sel_label = [0] * len(input_ids)
            for idx in range(targets_start, targets_end+1):
                sel_label[idx] = 1
                
            if self.qa:
                input_ids += [2] + [self.sent_id[sentiment]] + [2]
                token_type_ids += [0] * 3 #roberta does not use it during pretraining, can finetune
                mask += [1] * 3
                tweet_offsets += [(0,0)] * 3
                sel_label += [0] * 3
                
            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + ([1] * padding_length)
                mask = mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)
                tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
                sel_label = sel_label + [0] * padding_length
                
        return {
            'ids': input_ids[:self.max_length],
            'mask': mask[:self.max_length],
            'token_type_ids': token_type_ids[:self.max_length],
            'sel_label': sel_label[:self.max_length],
            'offsets': tweet_offsets[:self.max_length]
        }


    def __getitem__(self, index):
        assert self.current_fold is not None
        if self.test:
            try:
                text, sel_text, label = self.val_text[self.current_fold][index],\
                    self.val_sel_text[self.current_fold][index],\
                    self.val_label[self.current_fold][index]
            except:
                raise ValueError('k:{}, index:{}'.format(self.current_fold, index))
        else:
            text, sel_text, label = self.train_text[self.current_fold][index],\
                self.train_sel_text[self.current_fold][index],\
                self.train_label[self.current_fold][index]  
        
        inputs = self.tokenize_and_getSelLabel(text, sel_text, label)
        
        return torch.LongTensor(inputs['ids']), torch.LongTensor(inputs['mask']),\
            torch.LongTensor(inputs['sel_label']), torch.LongTensor([label]),\
            torch.LongTensor(inputs['token_type_ids']), inputs['offsets'], text, sel_text

    def __len__(self):
        if self.test:
            return len(self.val_text[self.current_fold])
        else:
            return len(self.train_text[self.current_fold])

