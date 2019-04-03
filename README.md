## Shake-Shake pytorch
This is a unofficial Shake-Shake regularization implement of pytorch
with Python3.

### Usage
For train a model, please just use train [scripts](./scripts)

Shake Shake model normally need to train 1800 epochs, but this repo just
train **220 epochs**, so if you need better results you could train a
lettle more. 

### Results on CIFAR-10 

Model        | Proposed | This repo|
-------      |:--------:|:--------:|
SSI 26 2x32d |3.55	    |Training  |
SSI 26 2x64d |2.98	    |Training  |
SSI 26 2x96d |2.86	    |Training  |
SSI 26 2x112d|2.82	    |Training  |
