#!/bin/bash

# Usage: eval_model_tasks.sh <model> <eval_tasks> [<iteration>]

iter_arg=""

eval_tasks="$2"
if [ $# -gt 2 ]; then
    iter_arg="--iteration $3"
fi

model=/mnt/ssd-1/igor/gpt-neox/models/$1
res_prefix=/mnt/ssd-1/igor/gpt-neox/results/$1

if [ $# -gt 2 ]; then
    res_prefix="$res_prefix.global_step$3"
fi

./deepy.py evaluate.py -d "$model/configs" config.yml --eval_results_prefix "$res_prefix" $iter_arg --eval_tasks $eval_tasks
