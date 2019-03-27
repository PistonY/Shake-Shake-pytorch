# -*- coding: utf-8 -*-
# Author: X.Yang
# Concat: pistonyang@gmail.com
# Date  : 3/27/19

import torch

__all__ = ['Accuracy']


class Accuracy(object):
    def __init__(self, name='Acc'):
        self.num_metric = 0
        self.num_inst = 0
        self.name = name

    def reset(self):
        self.num_metric = 0
        self.num_inst = 0

    def update(self, labels, preds):
        _, pred = torch.max(preds, dim=1)
        pred = pred.cpu().view(-1).detach().numpy().astype('int32')
        lbs = labels.cpu().view(-1).detach().numpy().astype('int32')
        self.num_metric += int((pred == lbs).sum())
        self.num_inst += len(lbs)

    def get(self):
        return self.name, self.num_metric / self.num_inst
