#!/usr/bin/env bash
python train.py --cutout --epochs 220 --batch-size 128 --depth 96 --width 32 --mixup --no-wd
