#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import time
from utils import Data, read_sessions, index_from_one, generate_mask
from model import *
import os

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_random_seed(2021)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=128, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=0, help='l2 penalty')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')

parser.add_argument('--alpha', type=float, default=1, help='')
parser.add_argument('--scale', type=float, default=9, help='')
opt = parser.parse_args()
print(opt)

def main():
    data_floder = '../datasets/'
    train_seqs = read_sessions(data_floder + opt.dataset + '/' + 'train.txt')
    test_seqs = read_sessions(data_floder + opt.dataset + '/' + 'test.txt')
    if opt.dataset == 'sample':
        test_seqs = train_seqs[int(len(train_seqs)*0.01) : int(len(train_seqs)*0.011)]
        train_seqs = train_seqs[:int(len(train_seqs)*0.01)]
        max_train = max(max(row) for row in train_seqs)
        max_test = max(max(row) for row in test_seqs)

    # index from 1
    train_seqs = index_from_one(train_seqs)
    test_seqs = index_from_one(test_seqs)

    print('num of training sessions = ', len(train_seqs))
    print('num of test sessions = ', len(test_seqs))
    train_lens = [len(seq) for seq in train_seqs]
    test_lens = [len(seq) for seq in test_seqs]
    print('max train session length = ', max(train_lens))
    print('max test session length = ', max(test_lens))

    with open(data_floder + opt.dataset + '/' + 'num_items.txt', 'r') as g:
        opt.n_node = int(g.readline()) + 1 # plus 1
    if opt.dataset == 'sample':
        # index from 0 -> index from 1 plus 1 and plus 1 for the padding 0
        opt.n_node = max(max_train, max_test) + 1 + 1
    
    train_data = Data(train_seqs, opt, shuffle=True)
    test_data = Data(test_seqs, opt, shuffle=False)

    model = trans_to_cuda(SessionGraph(opt, opt.n_node))
    structural_mask, temporal_mask, sasrec_mask = generate_mask(opt.batchSize, 20)
    opt.structural_mask = trans_to_cuda(structural_mask)
    opt.temporal_mask = trans_to_cuda(temporal_mask)
    opt.sasrec_mask = trans_to_cuda(sasrec_mask)

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data, opt)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
