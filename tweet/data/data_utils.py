import torch

def tweet_collate_fn(batch):
    ids, masks, sel_labels, labels = zip(*batch)
    return torch.stack(ids, dim=0), \
            torch.stack(masks, dim=0), \
            torch.stack(sel_labels, dim=0), \
            torch.stack(labels, dim=0)

def infer_collate_fn(batch):
    ids, masks, labels, ori_toks, df_ids = zip(*batch)
    return torch.stack(ids, dim=0), \
            torch.stack(masks, dim=0), \
            torch.stack(labels, dim=0), \
            ori_toks, \
            df_ids