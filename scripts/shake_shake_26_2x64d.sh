#!/usr/bin/env bash
python train.py --cutout --epochs 220 --batch-size 256 --depth 26 --width 64 --mixup --no-wd
