# -*- coding: utf-8 -*-
# Author: X.Yang
# Concat: pistonyang@gmail.com
# Date  : 3/27/19

import torch
from model.ss_resnet import ShakeResNet
from utils.summary import summary

model = ShakeResNet(26, 112, 10).cuda()
# print(net(torch.randn(1, 3, 32, 32).cuda()))
summary(model, (3, 32, 32))