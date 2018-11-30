#!/usr/bin/env bash

python ./train.py \
--checkpoint_path=../models/vgg16/vgg_16.ckpt  \
--output_dir=../output/models/  \
--dataset_train=../datasets/train/fcn_train.record  \
--dataset_val=../datasets/train/fcn_val.record \
--batch_size=16  \
--max_steps=1500  \
--learning_rate=0.0001  \

