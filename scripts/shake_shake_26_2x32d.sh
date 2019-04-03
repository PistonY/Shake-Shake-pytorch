#!/usr/bin/env bash
python train.py --cutout --epochs 220 --batch-size 256 --depth 26 --width 32 --mixup --no-wd