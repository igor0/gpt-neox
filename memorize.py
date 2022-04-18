#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Josh Levy-Kramer <josh@levykramer.co.uk>. All rights reserved.
# This file is based on code by the authors denoted below and has been modified from its original version.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from megatron.utils import print_rank_0, setup_for_inference_or_eval

from megatron.memorize.memorize_utils import memorize_from_file
from megatron.memorize.memindex import build_memindex
from megatron.neox_arguments import NeoXArgs

def main():
    """
    Generate text/sample model
    """
    if False:
        model, neox_args = setup_for_inference_or_eval()
        model.eval()

        print_rank_0(f'Memorizing from input file {neox_args.memorize_input_file}')
        if neox_args.memorize_input_file is None:
            raise ValueError(f"`memorize_input_file` is not specified.")

        memorize_from_file(
            neox_args=neox_args,
            model=model,
            input_file=neox_args.memorize_input_file,
        )
    else:
        neox_args = NeoXArgs.consume_neox_args()
        build_memindex(neox_args.memory_save, neox_args.attention_config)

if __name__ == "__main__":
    main()
