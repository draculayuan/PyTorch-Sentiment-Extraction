import torch

def tweet_collate_fn(batch):
    ids, masks, sel_labels, labels, type_ids, offsets, text, sel_text = zip(*batch)
    return torch.stack(ids, dim=0), \
            torch.stack(masks, dim=0), \
            torch.stack(sel_labels, dim=0), \
            torch.stack(labels, dim=0), \
            torch.stack(type_ids), \
            offsets, text, sel_text

def infer_collate_fn(batch):
    ids, masks, labels, type_ids, offsets, text, df_id = zip(*batch)
    return torch.stack(ids, dim=0), \
            torch.stack(masks, dim=0), \
            torch.stack(labels, dim=0), \
            torch.stack(type_ids, dim=0), \
            offsets, \
            text, df_id