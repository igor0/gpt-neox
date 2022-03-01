#!/usr/bin/env python
import argparse
import os

import xform

def canonicalize_args(args):
    args.orig_model_path = os.path.abspath(args.orig_model_path)
    args.new_model_path = os.path.abspath(args.new_model_path)
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="extra_linear", choices=['logit_lens', 'extra_linear', 'final_linear', 'final_norm', 'out_linear_all', 'in_linear_all', 'all', 'all_100k'])
    parser.add_argument("--extra_linear_only", action='store_true')
    parser.add_argument("orig_model_path")
    parser.add_argument("new_model_path")

    args = canonicalize_args(parser.parse_args())

    transform = xform.model_transform(args)
    transform.copy_head()

if __name__ == "__main__":
    main()
