import torch

def tweet_collate_fn(batch):
    ids, masks, sel_labels, labels, type_ids = zip(*batch)
    return torch.stack(ids, dim=0), \
            torch.stack(masks, dim=0), \
            torch.stack(sel_labels, dim=0), \
            torch.stack(labels, dim=0), \
            torch.stack(type_ids)

def infer_collate_fn(batch):
    ids, masks, labels, ori_toks, df_ids, type_ids = zip(*batch)
    return torch.stack(ids, dim=0), \
            torch.stack(masks, dim=0), \
            torch.stack(labels, dim=0), \
            ori_toks, \
            df_ids, \
            torch.stack(type_ids, dim=0)