import torch
import torch.nn as nn

def loc_loss(pred, seq_target):
    '''
    pred: bs x seq x 2
    seq_target: bs x seq
    '''
    criterion = nn.CrossEntropyLoss()
    start, end = pred[:, :, 0], pred[:, :, 1]
    # build target
    target_start = torch.LongTensor([]).cuda()
    target_end = torch.LongTensor([]).cuda()
    for i in range(seq_target.size(0)):
        temp = (seq_target[i] == 1).nonzero()
        target_start = torch.cat((target_start, temp[0]), dim = 0)
        target_end = torch.cat((target_end, temp[-1]), dim = 0)

    return criterion(start, target_start) + criterion(end, target_end)