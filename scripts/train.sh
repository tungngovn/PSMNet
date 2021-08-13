#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath dataset/ \
               --epochs 0 \
               --loadmodel ./trained/checkpoint_10.tar \
               --savemodel ./trained/

