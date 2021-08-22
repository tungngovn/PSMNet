#!/bin/bash

python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype apolloscape \
                   --datapath dataset/apolloscape/stereo_train/ \
                   --epochs 100 \
                   --batch_size 2 \
                   --loadmodel ./trained/finetune_18.tar \
                   --savemodel ./trained/ 2>&1 |tee ./logs/train_logs.txt

