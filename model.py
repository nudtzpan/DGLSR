#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from utils import trans_to_cuda, trans_to_cpu, l2_norm, cross_entropy_margin


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(self.opt.n_node, self.opt.hiddenSize)

        self.structural_q = nn.Linear(self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.structural_k = nn.Linear(self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.temporal_q = nn.Linear(self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.temporal_k = nn.Linear(self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.sasrec_q = nn.Linear(self.opt.hiddenSize, self.opt.hiddenSize, bias=True)
        self.sasrec_k = nn.Linear(self.opt.hiddenSize, self.opt.hiddenSize, bias=True)

        self.concat = nn.Linear(3*self.opt.hiddenSize, self.opt.hiddenSize, bias=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.opt.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def structural(self, time_hidden):
        # time_hidden: bs * seq_len * seq_len * latent_size (time, item)
        bs, seq_len = time_hidden.shape[0], time_hidden.shape[1]
        mask = self.opt.structural_mask[:bs, :seq_len, :seq_len, :seq_len]

        q, k, v = self.structural_q(time_hidden), self.structural_k(time_hidden), time_hidden
        sim = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.opt.hiddenSize) # bs * seq_len * seq_len
        sim = sim + mask
        sim = torch.softmax(sim, -1) # bs * seq_len * seq_len * seq_len (time, item, item)
        structural_hidden = torch.matmul(sim, v) # bs * seq_len * seq_len * latent_size (time, item)
        output = structural_hidden
        return output

    def temporal(self, structural_hidden):
        # structural_hidden: bs * seq_len * seq_len * latent_size (time, item)
        bs, seq_len = structural_hidden.shape[0], structural_hidden.shape[1]
        mask = self.opt.temporal_mask[:bs, :seq_len, :seq_len, :seq_len]    

        # q, k, v: bs * seq_len * seq_len * latent_size (item, time)
        q, k, v = self.temporal_q(structural_hidden.transpose(2, 1)), \
                  self.temporal_k(structural_hidden.transpose(2, 1)), structural_hidden.transpose(2, 1)
        sim = torch.matmul(q, k.transpose(-2, -1)) # bs * seq_len * seq_len * seq_len (item, time, time)
        sim = sim + mask
        sim = torch.softmax(sim, -1) # bs * seq_len * seq_len * seq_len (item, time, time)
        temp_hidden = torch.matmul(sim, v) # bs * seq_len * seq_len * latent_size (item, time)
        temporal_hidden = temp_hidden.transpose(2, 1) # bs * seq_len * seq_len * latent_size (time, item)
        output = temporal_hidden
        return output

    def sasrec(self, temporal_hidden):
        # temporal_hidden: bs * seq_len * seq_len * latent_size (time, item)
        bs, seq_len = temporal_hidden.shape[0], temporal_hidden.shape[1]
        mask = self.opt.sasrec_mask[:bs, :seq_len, :seq_len, :seq_len]

        q, k, v = self.sasrec_q(temporal_hidden), self.sasrec_k(temporal_hidden), temporal_hidden
        sim = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.opt.hiddenSize) # bs * seq_len * seq_len * seq_len (time, item, item)
        sim = sim + mask
        sim = torch.softmax(sim, -1) # bs * seq_len * seq_len * seq_len (time, item, item)
        sat_hidden = torch.matmul(sim, v) # bs * seq_len * seq_len * latent_size (time, item)
        output = sat_hidden
        return output

    def forward(self, inputs):
        # inputs: bs * seq_len
        seq_len = inputs.shape[1]
        initial_hidden = self.embedding(inputs) # bs * seq_len * latent_size
        hidden = initial_hidden.unsqueeze(1).repeat(1, seq_len, 1, 1) # bs * seq_len * seq_len * latent_size (time, item)
        hidden_combine = []

        hidden = self.structural(hidden) # bs * seq_len * seq_len * latent_size (time, item)
        structural_eye = hidden[:, torch.arange(seq_len).long(), torch.arange(seq_len).long(), :]
        hidden_combine.append(structural_eye)

        hidden = self.temporal(hidden) # bs * seq_len * seq_len * latent_size (time, item)
        temporal_eye = hidden[:, torch.arange(seq_len).long(), torch.arange(seq_len).long(), :]
        hidden_combine.append(temporal_eye)

        hidden = self.sasrec(hidden) # bs * seq_len * seq_len * latent_size (time, item)
        sasrec_eye = hidden[:, torch.arange(seq_len).long(), torch.arange(seq_len).long(), :]
        hidden_combine.append(sasrec_eye)

        a = self.concat(torch.cat(hidden_combine, -1))

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        a, b = l2_norm(a), l2_norm(b)
        scores = torch.matmul(a, b.transpose(1, 0))

        return scores

def forward(model, i, data):
    inputs, targets = data.get_slice(i)
    inputs = trans_to_cuda(torch.Tensor(inputs).long())
    scores = model(inputs)

    scores = scores.view(-1, scores.shape[-1])
    targets = np.reshape(targets, (-1,))
    return targets, scores

def train_test(model, train_data, test_data, opt):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.opt.batchSize)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        mask = torch.sign(targets)
        loss = cross_entropy_margin(scores, targets - 1, mask, opt)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.opt.batchSize)
    test_mask = []
    for i, j in zip(slices, np.arange(len(slices))):
        targets, scores = forward(model, i, test_data)
        test_mask = test_mask + np.sign(targets).tolist()
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target in zip(sub_scores, targets):
            # +0: transform True/False into 1/0
            hit.append(np.isin(target - 1, score)+0)
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    test_mask = np.array(test_mask)
    hit = np.sum(hit*test_mask) / np.sum(test_mask) * 100
    mrr = np.sum(mrr*test_mask) / np.sum(test_mask) * 100
    return hit, mrr
