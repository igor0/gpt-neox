"""
Tool to slim down PyTorch checkpoints.

Accepts two arguments: input_dir and output_dir. Loads each checkpoint in
directory input_dir, detaches and clones each tensor, and writes the modified
checkpoint into output_dir directory.
"""

import argparse
import os
import shutil
import sys
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Directory containing checkpoints to slim down')
    parser.add_argument('output_dir', help='Directory to write slimmed checkpoints to')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for filename in os.listdir(args.input_dir):
        if not filename.endswith('.pt'):
            continue
        print('Slimming down {}'.format(filename))
        checkpoint = torch.load(os.path.join(args.input_dir, filename))
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                checkpoint[key] = value.detach().clone()
            if isinstance(value, dict) and key == "optimizer":
                for key2 in ["fp32_groups_flat", "optimizer_state_dict"]:
                    if key2 in value:
                        del value[key2]
        torch.save(checkpoint, os.path.join(args.output_dir, filename))

if __name__ == '__main__':
    main()

