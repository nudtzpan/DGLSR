#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import numpy as np
import torch
import pandas as pd


def cross_entropy_margin(x, tar_idx, mask, opt):
    scores = x.detach()
    tar_score = scores[torch.arange(x.shape[0]).long(), tar_idx] # bs
    margin = scores - tar_score.unsqueeze(-1) # bs * candidate_num
    margin = torch.sigmoid(margin)
    x = x + margin * opt.alpha

    x = torch.log_softmax(x*opt.scale, -1) # bs * candidate_num
    loss_temp = x[torch.arange(x.shape[0]).long(), tar_idx] # bs

    loss = -torch.sum(loss_temp*mask) / torch.sum(mask) # 1
    return loss

def generate_structural_mask(bs, seq_len):
    triu_1 = 1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    triu_1 = triu_1.unsqueeze(-2).repeat(1, triu_1.shape[1], 1) * triu_1.unsqueeze(-1).repeat(1, 1, triu_1.shape[1])
    triu_2 = torch.triu(torch.ones((seq_len, seq_len)), diagonal=2)
    triu_2 = 1 - triu_2 - triu_2.transpose(-2, -1) # - torch.eye(seq_len)
    triu_2 = triu_2.unsqueeze(0).repeat(seq_len, 1, 1)

    mask = triu_1 * triu_2
    mask = (1-mask) * -10000
    mask = mask.unsqueeze(0).repeat(bs, 1, 1, 1) # bs * seq_len * seq_len * seq_len
    return mask

def generate_temporal_mask(bs, seq_len):
    triu_1 = torch.triu(torch.ones(seq_len, seq_len), diagonal=0)
    triu_1 = triu_1.unsqueeze(-2).repeat(1, triu_1.shape[1], 1) * triu_1.unsqueeze(-1).repeat(1, 1, triu_1.shape[1])
    triu_1 = triu_1.unsqueeze(0).repeat(bs, 1, 1, 1)
    triu_2 = 1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1) # seq_len * seq_len
    triu_2 = triu_2.unsqueeze(0).repeat(seq_len, 1, 1).unsqueeze(0).repeat(bs, 1, 1, 1)

    mask = triu_1 * triu_2
    mask = (1-mask) * -10000 # bs * seq_len * seq_len * seq_len
    return mask

def generate_sasrec_mask(bs, seq_len):
    attn_shape = (seq_len, seq_len)
    triu_1 = 1 - torch.triu(torch.ones(attn_shape), diagonal=1)
    triu_1 = triu_1.unsqueeze(0).repeat(bs, 1, 1).unsqueeze(1).repeat(1, seq_len, 1, 1) # bs * seq_len * seq_len * seq_len
    triu_2 = 1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    triu_2 = triu_2.unsqueeze(-2).repeat(1, triu_2.shape[1], 1) * triu_2.unsqueeze(-1).repeat(1, 1, triu_2.shape[1])

    mask = triu_1 * triu_2
    mask = (1-mask) * -10000 # bs * seq_len * seq_len * seq_len
    return mask

def generate_mask(bs, seq_len):
    structural_mask = generate_structural_mask(bs, seq_len)
    temporal_mask = generate_temporal_mask(bs, seq_len)
    sasrec_mask = generate_sasrec_mask(bs, seq_len)
    return structural_mask, temporal_mask, sasrec_mask

def l2_norm(x):
    y = x / torch.sqrt(torch.sum(x**2, -1, keepdim=True) + 1e-24)
    return y

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    sessions = (sessions).tolist()
    return sessions

def index_from_one(seqs):
    for i in range(len(seqs)):
        seq = seqs[i]
        new_seq = [item+1 for item in seq]
        seqs[i] = new_seq
    return seqs

def seq_augument(seqs):
    aug_seqs = []
    for seq in seqs:
        for i in range(2, len(seq)+1):
            aug_seqs.append(seq[:i])
    return aug_seqs

def inputs_target_split(seqs):
    inputs, targets = [], []
    for seq in seqs:
        inputs.append(seq[:-1])
        targets.append(seq[-1])
    return [inputs, targets]

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    return us_pois

class Data():
    def __init__(self, data, opt, shuffle=False):
        sess_num = len(data)
        inputs, targets = [], []
        for seq in data:
            inputs.append(seq[:-1])
            targets.append(seq[1:])

        max_len = max([len(seq) for seq in inputs])
        for i in range(len(inputs)):
            inputs[i] = inputs[i] + [0]*(max_len-len(inputs[i]))
        for i in range(len(targets)):
            targets[i] = targets[i] + [0]*(max_len-len(targets[i]))

        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(targets)
        self.length = sess_num
        self.shuffle = shuffle
        self.opt = opt

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, targets = self.inputs[i], self.targets[i]

        max_len = max(np.sum(np.sign(inputs), -1))
        inputs = inputs[:, :max_len]
        targets = targets[:, :max_len]

        return inputs, targets
