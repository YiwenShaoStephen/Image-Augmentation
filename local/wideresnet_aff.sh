#!/bin/bash
# cmd=queue.pl
dataset=cifar100
layers=28
width=10

zoom_prob=0.5
zoom_low=0.7
zoom_upp=1.4

rotation_prob=0.5
rotation_left=-30
rotation_right=30

stretch_prob=0.5
stretch_low=0.33
stretch_upp=3

. ./parse_options.sh

name=resnet_${layers}_${width}_r${rotation_prob}_${rotation_left}to${rotation_right}_z${zoom_prob}_${zoom_low}to${zoom_upp}_s${stretch_prob}_${stretch_low}to${stretch_upp}

CUDA_VISIBLE_DEVICES=$(free-gpu) /home/yshao/miniconda2/bin/python ./train_wide.py \
		    --dataset $dataset --layers $layers --widen-factor $width \
		    --name $name \
		    --zoom-prob $zoom_prob \
		    --zoom-range $zoom_low $zoom_upp \
		    --rotation-prob $rotation_prob \
		    --rotation-degree $rotation_left $rotation_right \
		    --stretch-prob $stretch_prob \
		    --stretch-range $stretch_low $stretch_upp
