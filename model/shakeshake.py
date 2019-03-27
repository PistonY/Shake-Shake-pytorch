# -*- coding: utf-8 -*-
# Author: X.Yang
# Concat: pistonyang@gmail.com
# Date  : 3/26/19

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.autograd.function import Function


class ShakeShake(Function):
    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.cuda.FloatTensor(x1.size(0)).uniform_()
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_outputs):
        beta = torch.cuda.FloatTensor(grad_outputs.size(0)).uniform_()
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_outputs)
        beta = Variable(beta)
        return beta * grad_outputs, (1 - beta) * grad_outputs, None


class Shortcut(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super(Shortcut, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        h = F.relu(x)

        h1 = F.avg_pool2d(h, 1, self.stride)
        h1 = self.conv1(h1)

        h2 = F.avg_pool2d(F.pad(h, [-1, 1, -1, 1]), 1, self.stride)
        h2 = self.conv2(h2)

        h = torch.cat((h1, h2), 1)
        return self.bn(h)
