#!/bin/bash
#########################################################################
# File Name: run.sh
# Author: daniel.wang
# Mail: wangzhanoop@163.com
# Created Time: Sun 27 Aug 2017 10:52:49 AM CST
# Brief: 
#########################################################################

one_run() {
style=$1
sws=$2
tw=$3
ssize=$4
name="${style}-1X${sws}"
rm -rf models/${name}
python train.py --style_image style_images/${style}.jpg \
	--naming ${name} \
	--style ${style} \
	--content_weight 1.0 \
	--style_weight 1.0 \
	--tv_weight ${tw} \
	--content_layers "vgg_16/conv3/conv3_3" \
	--style_layers "vgg_16/conv1/conv1_2,vgg_16/conv2/conv2_2,vgg_16/conv3/conv3_3,vgg_16/conv4/conv4_3" \
	--style_layers_weights ${sws} \
	--style_size ${ssize} \
	--epoch 2 \
	--max_iter 1000000 \
	--device "0"

python test.py --model_file "models/${name}/fast-style-model-done" \
	--image_file "chicago.jpg" \
	--save_file "${name}.jpg" \
	--device "0"
}

one_run polynesia "70,70,70,70" 0 960

