#!/bin/bash

for i in {2..32}; do cat - > layer$i.yml <<EEE
{
    "pythia_num_layers": $i,
    "load": "/mnt/ssd-1/igor/gpt-neox/models/dense_small_checkpoints/global_step485000/shrunk$i",
    "save": "/mnt/ssd-1/igor/gpt-neox/models/dense_small_checkpoints/global_step485000/shrunk$i",
}
EEE
done
