## Shake-Shake pytorch
This is a unofficial Shake-Shake regularization implement of pytorch
with Python3.

And boost much better than proposed using less time.
### Usage
For train a model, please just use train [scripts](./scripts)

Shake Shake model normally need to train 1800 epochs, but this repo just
train **220 epochs**, so if you need better results you could train a
lettle more. 

Pretrained model/log(s) is [here](https://drive.google.com/open?id=15P7fZ7Qf2biIIjo2lNPj7W3yx-2qnZjD)

### Results on CIFAR-10(error rate)

Model        | Proposed | This repo|Model size| Improved |
-------      |:--------:|:--------:|:--------:|:--------:|
epochs       |1800      |220       |-         |(save time) 818%|
SSI 26 2x32d |**3.55**  |3.91      |2.9M     |-0.37%    |
SSI 26 2x64d |2.98	    |**2.90**  |11.7M    |0.08%     |
SSI 26 2x96d |2.86	    |**2.66**  |26.2M    |0.20%     |
SSI 26 2x112d|2.82	    |**2.56**  |35.6M    |0.26%     |
