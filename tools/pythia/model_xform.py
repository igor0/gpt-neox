#!/usr/bin/env python3
import argparse
import os

import xform

def canonicalize_args(args):
    args.orig_model_path = os.path.abspath(args.orig_model_path)
    args.new_model_path = os.path.abspath(args.new_model_path)
    if args.head is not None:
        args.head = os.path.abspath(args.head)
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="extra_linear", choices=['logit_lens', 'extra_linear', 'final_linear', 'final_norm', 'out_linear_all', 'in_linear_all', 'all', 'all_100k', 'pre_logit_lens', 'pre_extra_linear'])
    parser.add_argument("--head", type=str)
    parser.add_argument("--predict", type=str, choices=['self', 'abs', 'abslog', 'abssqrt', 'prev', 'sink', 'next2', 'prev2'])
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--masterport", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_gpus", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--random_init", action='store_true')
    parser.add_argument("orig_model_path")
    parser.add_argument("new_model_path")

    args = canonicalize_args(parser.parse_args())

    transform = xform.model_transform(args)

    layers_num = args.num_layers if args.num_layers is not None else transform.orig.layers_num
    mutable = transform.link_new_model(layers_num)
    mutable.modify_layer_checkpoint()
    mutable.modify_optimizer_checkpoint()

if __name__ == "__main__":
    main()
