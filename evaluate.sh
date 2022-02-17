#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,7
source ~/igor/gpt-neox/venv/bin/activate

python ./evaluate.py "$@"
