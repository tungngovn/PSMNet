#!/bin/bash

python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype apolloscape \
                   --datapath dataset/apolloscape/stereo_train/ \
                   --epochs 100 \
                   --batch_size 4 \
                   --loadmodel ./trained/pretrained_sceneflow.tar \
                   --savemodel ./trained/

