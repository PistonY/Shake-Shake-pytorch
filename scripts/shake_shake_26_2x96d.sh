#!/usr/bin/env bash
python ../train.py --cutout --epochs 220 --batch-size 128 --depth 26 --width 96 --mixup --no-wd
