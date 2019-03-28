# -*- coding: utf-8 -*-
# Author: X.Yang
# Concat: pistonyang@gmail.com
# Date  : 3/27/19

import time
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10

from torchmetric.metric import Accuracy, Loss as mloss
from PIL import Image
from model.ss_resnet import ShakeResNet
from utils import Cutout, mixup_data, mixup_criterion, IterLRScheduler, split_weights

use_cutout = True
cutout = Cutout() if use_cutout else None
normalizer = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                  std=[0.2471, 0.2435, 0.2616])


def transformer(im):
    im = np.array(im)
    im = np.pad(im, pad_width=((4, 4), (4, 4), (0, 0)), mode='constant')
    if use_cutout:
        im = cutout(im)
    im = Image.fromarray(im)
    auglist = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
    ])
    im = auglist(im)
    return im


def trans_test(im):
    auglist = transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ])
    im = auglist(im)
    return im


batch_size = 256
train_data = DataLoader(
    CIFAR10('/media/piston/data/pytorch/dataset/', transform=transformer, train=True),
    batch_size, shuffle=True, num_workers=4, drop_last=True)
val_data = DataLoader(
    CIFAR10('/media/piston/data/pytorch/dataset/', transform=trans_test, train=False),
    batch_size, num_workers=4)

num_classes = 10
model = ShakeResNet(26, 64, num_classes).cuda()


def test(epoch=0, save_stat=True):
    model.eval()
    acc_metric = Accuracy()
    with torch.no_grad():
        for images, labels in val_data:
            images = images.cuda()
            labels = labels.cuda().long()
            output = model(images)
            acc_metric.update(labels, output)
    met_name, acc = acc_metric.get()
    test_str = 'Epoch {}. Test {}: {:.5f}'.format(epoch, met_name, acc)
    print(test_str)
    if save_stat:
        torch.save(model.state_dict(), '{}/epoch_{}_{:.5}.pkl'.format('param', epoch, acc))


epochs = 220
base_lr = 0.1 * (batch_size // 64)
num_train_samples = 50000
mixup = True

num_iterations = len(train_data) * epochs
lr_warmup_iters = len(train_data) * 5

lr_scheduler = IterLRScheduler(mode='cosine', baselr=base_lr, niters=num_iterations, warmup_iters=lr_warmup_iters)


def train():
    params = split_weights(model)
    optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    Loss = nn.CrossEntropyLoss()
    metric_loss = mloss()
    alpha = 1. if mixup else 0.
    iterations = 0
    for epoch in range(epochs):
        model.train()
        metric_loss.reset()
        st_time = time.time()
        if mixup and epoch > epochs - 20:
            alpha = 0.
        for i, (trans, labels) in enumerate(train_data):
            trans, targets_a, targets_b, lam = mixup_data(trans.cuda(), labels.cuda(), alpha=alpha)
            trans, targets_a, targets_b = map(Variable, (trans, targets_a, targets_b))
            # trans = Variable(trans.cuda())
            # labels = Variable(labels.cuda()).long()
            outputs = model(trans)
            loss = mixup_criterion(Loss, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric_loss.update(loss)
            iterations += 1
            lr_scheduler.update(optimizer, iterations)
        learning_rate = lr_scheduler.get()
        met_name, metric = metric_loss.get()
        epoch_time = time.time() - st_time
        epoch_str = 'Epoch {}. Train {}: {:.5f}. {} samples/s. lr {:.5}'. \
            format(epoch, met_name, metric, int(num_train_samples // epoch_time), learning_rate)
        print(epoch_str)
        test(epoch, False)


if __name__ == '__main__':
    train()
