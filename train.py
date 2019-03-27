# -*- coding: utf-8 -*-
# Author: X.Yang
# Concat: pistonyang@gmail.com
# Date  : 3/27/19

import time
import numpy as np
from torch import nn, optim
from torch.autograd import Variable

from torchmetric.metric import Accuracy
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from model.ss_resnet import ShakeResNet
from utils.cutout import Cutout


def _read_files(path):
    img = Image.open(path)
    return img


cutout = Cutout()
normalizer = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                  std=[0.2471, 0.2435, 0.2616])


def transformer(im):
    im = np.array(im)
    im = np.pad(im, pad_width=((4, 4), (4, 4), (0, 0)), mode='constant')
    im = Image.fromarray(cutout(im))
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


batch_size = 64
train_data = DataLoader(
    CIFAR10('/media/piston/data/pytorch/dataset/', transform=transformer, train=True),
    batch_size, shuffle=True, num_workers=4, drop_last=True)
val_data = DataLoader(
    CIFAR10('/media/piston/data/pytorch/dataset/', transform=trans_test, train=False),
    batch_size, num_workers=4)

num_classes = 10
model = ShakeResNet(26, 112, num_classes).cuda()


def test():
    model.eval()
    acc_metric = Accuracy()
    for images, labels in val_data:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda()).long()
        output = model(images)
        acc_metric.update(labels, output)
    met_name, acc = acc_metric.get()
    test_str = 'Test {}: {:.5f}'.format(met_name, acc)
    print(test_str)


epochs = 100
lr = 0.1
num_train_samples = 50000


def train():
    global lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    Loss = nn.CrossEntropyLoss()
    metric_acc = Accuracy()
    for epoch in range(epochs):
        model.train()
        metric_acc.reset()
        st_time = time.time()
        for i, (trans, labels) in enumerate(train_data):
            trans = Variable(trans.cuda())
            labels = Variable(labels.cuda()).long()
            optimizer.zero_grad()
            outputs = model(trans)
            metric_acc.update(labels, outputs)
            loss = Loss(outputs, labels)
            loss.backward()
            optimizer.step()
        met_name, acc = metric_acc.get()
        epoch_time = time.time() - st_time
        epoch_str = 'Train {}: {:.5f}. {} samples/s.'.format(met_name, acc, int(num_train_samples // epoch_time))
        print(epoch_str)
        if epoch in [int(e * epochs) for e in (0.3, 0.6, 0.9)]:
            lr /= 10
            print('reset learning rate to:', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print(param_group['lr'])
        test()
        # torch.save(model.state_dict(), '{}/{}.pkl'.format('param', epoch))


if __name__ == '__main__':
    train()
