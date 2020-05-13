import os
import logging
import shutil
import torch
import random
import copy
from datetime import datetime
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import math
import numpy as np
from collections import defaultdict
#import linklink as link

def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(ranks=rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank//group_size]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val):
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count

class IterLRScheduler(object):
    def __init__(self, optimizer, milestones, lr_mults, last_iter=-1):
        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestone, lr_mults)
        self.milestones = milestones
        self.lr_mults = lr_mults
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        for i, group in enumerate(optimizer.param_groups):
            if 'lr' not in group:
                raise KeyError("param 'lr' is not specified "
                               "in param_groups[{}] when resuming an optimizer".format(i))
        self.last_iter = last_iter

    def _get_lr(self):
        try:
            pos = self.milestones.index(self.last_iter)
        except ValueError:
            return list(map(lambda group: group['lr'], self.optimizer.param_groups))
        except:
            raise Exception('wtf?')
        return list(map(lambda group: group['lr']*self.lr_mults[pos], self.optimizer.param_groups))

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        world_size (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within world_size.
    """

    def __init__(self, dataset, world_size=None, rank=None, round_up=True):
        if world_size is None:
            world_size = link.get_world_size()
        if rank is None:
            rank = link.get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.round_up = round_up
        self.epoch = 0

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        if self.round_up:
            self.total_size = self.num_samples * self.world_size
        else:
            self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        if self.round_up or (not self.round_up and self.rank < self.world_size-1):
            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

        
class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1):
        if world_size is None:
            world_size = link.get_world_size()
        if rank is None:
            rank = link.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter*self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg+self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        #return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size

class GivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, last_iter=-1):
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.last_iter = last_iter

        self.total_size = self.total_iter*self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        all_size = self.total_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        #print('Warinihg! Shuffle for train is turned off in utils.py!!!!!\n\n')

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        #return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size
    
class GNNSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, last_iter=-1, sample_freq=2):
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.last_iter = last_iter
        self.sample_freq = sample_freq

        self.total_size = self.total_iter*self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            idx = self.indices[(self.last_iter+1)*self.batch_size:]
            return iter(idx)
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")
    def gen_idx_from_dict(self, dict_, seed, sample_freq):
        d = copy.deepcopy(dict_)
        idx = []
        while len(d.keys()) > 0:
            keys = list(d)
            random.seed(seed)
            random.shuffle(keys)
            for key in keys:
                for f in range(sample_freq):
                    idx.append(d[key][0])
                    d[key].pop(0)
                    if len(d[key]) == 0:
                        del d[key]
                        if f != (sample_freq-1):
                            break     

        return idx
    def gen_new_list(self):

        id_list = defaultdict(list)
        for i, id_ in enumerate(self.dataset.label_id):
            id_list[id_].append(i)
        rdm_seed = 0
        indices = self.gen_idx_from_dict(id_list, rdm_seed, self.sample_freq)
        rdm_seed += 1
        if len(indices) != len(self.dataset):
            raise ValueError('length of dataset doesnt match with length of sampled index')
        all_size = self.total_size
        indices = np.array(indices)
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1
        for j in range(num_repeat):
            indices = np.hstack([indices, self.gen_idx_from_dict(id_list, rdm_seed, 2)])
            rdm_seed += 1
        indices = indices[:all_size]

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        #return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size
    
def save_checkpoint(state, is_best, filename):
    torch.save(state, filename+'.pth.tar')
    if is_best:
        shutil.copyfile(filename+'.pth.tar', filename+'_best.pth.tar')

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_state(path, model, optimizer=None, test=False):
    #print('Notice!! Critic is not loaded here! \n')
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        if test is True:
            checkpoint['state_dict'] = remove_prefix(checkpoint['state_dict'], 'module.')
        pop_list = []
        for k in checkpoint['state_dict'].keys():
            if 'critic' in k:
                pop_list.append(k)
        for k in pop_list:
            checkpoint['state_dict'].pop(k, None)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        ckpt_keys = set(checkpoint['state_dict'].keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print('caution: missing keys from checkpoint {}: {}'.format(path, k))

        if optimizer != None:
            last_iter = checkpoint['step']
            optimizer.load_state_dict(checkpoint['optimizer'])
            #print("=> also loaded optimizer from checkpoint '{}' (iter {})"
            #      .format(path, last_iter))
            return last_iter
    else:
        print("=> no checkpoint found at '{}'".format(path))
        
def load_state_critic(path, model):
    #print('Notice!! Critic is not loaded here! \n')
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        mapping = {0: 'state_dict_pre', 1: 'state_dict_rec', 2: 'state_dict_f1'}
        for i, m in enumerate(model):
            m.load_state_dict(checkpoint[mapping[i]], strict=False)
            ckpt_keys = set(checkpoint[mapping[i]].keys())
            own_keys = set(m.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print('caution: missing keys from checkpoint {}: {}'.format(path, k))
    else:
        print("=> no checkpoint found at '{}'".format(path))
