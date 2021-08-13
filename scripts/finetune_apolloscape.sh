#!/bin/bash

python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype apolloscape \
                   --datapath dataset/apolloscape/stereo_train/ \
                   --epochs 100 \
                   --loadmodel ./trained/checkpoint_10.tar \
                   --savemodel ./trained/

