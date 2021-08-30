#!/bin/bash

python eval.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype apolloscape \
                   --datapath dataset/apolloscape/stereo_test/ \
                   --epochs 100 \
                   --batch_size 1 \
                   --loadmodel ./trained/finetune_50.tar \
                   --savemodel ./trained/ #2>&1 |tee ./logs/train_logs.txt

