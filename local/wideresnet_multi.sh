#!/bin/bash
# cmd=queue.pl
dataset=cifar100
layers=28
width=10
num_rotation_classes=12
epochs=200
rotation_prob=0.5
start=
resume=

. ./parse_options.sh

name=resnet_${layers}_${width}_multirotate_class${num_rotation_classes}

CUDA_VISIBLE_DEVICES=$(free-gpu) /home/yshao/miniconda2/bin/python \
		    ./train_wide_multi.py \
		    --dataset $dataset --layers $layers --widen-factor $width \
		    --name $name \
		    --num-rotate-classes $num_rotation_classes \
		    --rotation-prob $rotation_prob \
		    --num-rotate-classes $num_rotation_classes \
		    --epochs $epochs --start-epoch $start --resume $resume 
