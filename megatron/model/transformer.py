# coding=utf-8
#
# Copyright 2021 Biderman et al. This file is based on code by the authors denoted below and has been modified from its original version.
#
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

"""Transformer."""

import math
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn

from .norms import get_norm

from einops import rearrange
from pathlib import Path
from megatron import memorize
from megatron import mpu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.utils import exists
from megatron.model.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_torch,
    AliBi,
)
from megatron.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from megatron.model.utils import configure_sparse_attention
from megatron.memorize.paths import get_mem_dump_path
from megatron.memorize import load_memory_snapshot, index_memory_snapshot

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmasked-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmasked-attention-scores, attention-mask)
"""


class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self, neox_args, init_method, output_layer_init_method, parallel_output=False
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.bias_gelu_fusion = neox_args.bias_gelu_fusion

        # auto scale so geglu has equal parameters
        ff_mult = 4 * 2 / 3 if self.activation_type == "geglu" else 4
        ff_dim = (
            int(ff_mult * neox_args.hidden_size) * 2
            if self.activation_type == "geglu"
            else ff_mult * neox_args.hidden_size
        )
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )
        ff_dim_in = ff_dim // 2 if self.activation_type == "geglu" else ff_dim
        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim_in,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if (
            self.activation_type == "gelu" and self.bias_gelu_fusion
        ) or self.activation_type == "geglu":
            intermediate_parallel = self.activation_func(
                intermediate_parallel, bias_parallel
            )
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel + bias_parallel
            )

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class ParallelLinear(nn.Module):
    """
    A Parallel Linear Layer transforming the transformer outputs from hidden_size -> vocab_size
    """

    def __init__(
        self,
        neox_args,
        parallel_output=True,
        inference=False,
        init_method=nn.init.xavier_normal_,
    ):
        super().__init__()
        parallelism = neox_args.output_layer_parallelism
        if parallelism == "column":
            self.final_linear = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.padded_vocab_size,
                bias=False,
                init_method=init_method,
                gather_output=not parallel_output,
                skip_bias_add=False,
            )
        else:
            self.final_linear = mpu.RowParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.padded_vocab_size,
                bias=False,
                input_is_parallel=False,
                init_method=init_method,
                parallel_output=False if inference else parallel_output,
                skip_bias_add=False,
            )

    def forward(self, hidden_states):
        return self.final_linear(hidden_states)


class ParallelSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        get_key_value=False,
        parallel_output=False,
    ):
        super().__init__()

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.get_key_value = get_key_value
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            neox_args.hidden_size, neox_args.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            neox_args.num_attention_heads, world_size
        )
        self.pos_emb = neox_args.pos_emb
        self.memorize_mode = neox_args.memorize_mode

        # Strided linear layer.
        self.query_key_value = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=3 * neox_args.hidden_size,
            gather_output=False,
            init_method=init_method,
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        self.rpe = rpe

        if self.pos_emb == "alibi":
            self.alibi_embed = AliBi(
                neox_args.num_attention_heads,
                neox_args.model_parallel_size,
                mpu.get_model_parallel_rank(),
            )

        self.attention_type = neox_args.attention_config[layer_number]

        # TODO: this arg shouldn't need to be passed in - get from neox_args
        if rotary:
            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * neox_args.rotary_pct
                )
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            self.rotary_emb = RotaryEmbedding(
                dim, base=neox_args.rotary_emb_base, precision=neox_args.params_dtype
            )
        else:
            self.rotary_emb = None

        self.sparse = not self.attention_type.startswith("global") and not self.is_knn()
        if self.sparse:
            self.sparse_attn = configure_sparse_attention(
                neox_args,
                self.attention_type,
                self.num_attention_heads_per_partition,
                mpu=mpu,
            )
        else:
            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                input_in_fp16=self.fp16,
                input_in_bf16=self.bf16,
                upper_triang_mask_fusion=neox_args.scaled_upper_triang_masked_softmax_fusion,
                general_mask_fusion=neox_args.scaled_masked_softmax_fusion,
                mask_func=self.attention_mask_func,
                softmax_in_fp32=self.attention_softmax_in_fp32,
                scale=coeff,
            )

            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            self.attention_dropout = nn.Dropout(neox_args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )

        if self.is_knn():
            self.memory_kq_normalize = neox_args.memory_kq_normalize
            self.memory_attn_mode = neox_args.memory_attn_mode

            device = torch.cuda.current_device()

            if self.memorize_mode == "save":
                Path(neox_args.memory_save).mkdir(exist_ok = True, parents = True)
                def init_dumper(training):
                    if training:
                        # When memory_save is set, we should only dump during eval, not during
                        # training.
                        return None

                    mem_file = get_mem_dump_path(neox_args.memory_save, layer_number)
                    return memorize.MemoryDumper(
                        layer_number = layer_number,
                        file_path = mem_file,
                        dim = self.hidden_size_per_attention_head,
                        heads = self.num_attention_heads_per_partition,
                        index_memory_func = index_memory_snapshot
                    )
                memory_dumper_init = init_dumper
            else:
                memory_dumper_init = None

            if self.memorize_mode == "load":
                # Load precomputed memories from the specified index
                self.memory_snap = load_memory_snapshot(neox_args.memory_load, layer_number)
                self.memory_train = None
            else:
                # Maintain a sliding window of memories
                self.memory_train = memorize.MemoryLive(
                    device,
                    neox_args.memory_size,
                    neox_args.memory_invalid_query_mode,
                    memory_dumper_init = memory_dumper_init)
                self.memory_snap = None

            if neox_args.memory_attn_mode == "sigmoid":
                self.combine_attn_output_gate = nn.Parameter(0.002 * torch.ones(neox_args.num_attention_heads, 1, 1))

            if "memory_key_bias_48" in neox_args.memory_flags:
                self.memory_key_bias = torch.nn.Parameter(torch.rand(neox_args.num_attention_heads, self.hidden_size_per_attention_head, dtype=torch.float16).cuda()/48)
            else:
                self.memory_key_bias = None

            # [b, np, sq, sk]
            penalty_value = -10
            penalty_shape = (1, neox_args.num_attention_heads, 1, 1)
            self.memory_penalty_factor = 10
            self.memory_penalty = torch.nn.Parameter(torch.full(penalty_shape, penalty_value / self.memory_penalty_factor))

    def attention(
        self, query_layer, key_layer, value_layer, layer_past, attention_mask, penalty=None
    ):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
            attention_scores += rpe  # [1, np, sq, sk]

        if self.pos_emb == "alibi":
            attention_scores = self.alibi_embed(attention_scores)

        if penalty != None:
            attention_scores += penalty

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # [b * np, sq, sk] * [b * np, sk, hn] -> [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def sparse_attention(self, query_layer, key_layer, value_layer, attention_mask):
        # TODO: sparse attn dropout?
        # TODO: pad to block size
        # shape of q/k/v is [sq, b, np, hn] and needs to be transposed to [b, np, sq, hn]
        query_layer, key_layer, value_layer = map(
            lambda t: t.permute(1, 2, 0, 3).contiguous(),
            (query_layer, key_layer, value_layer),
        )
        # output shape [b, np(heads), sq, hn]
        attn_mask = attention_mask.to(query_layer.dtype) * -10000
        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
        else:
            rpe = None
        return self.sparse_attn(
            query_layer, key_layer, value_layer, attn_mask=attn_mask, rpe=rpe
        )

    def hidden_to_qkv(self, hidden_states, qkv):
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = qkv(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
            mixed_x_layer, 3
        )
        return query_layer, key_layer, value_layer

    def forward(self, hidden_states, attention_mask, eod_markers, layer_past=None):
        # hidden_states: [sq, b, h]
        # layer_past: [kv, sk, b, np, hn]

        # =====================
        # Query, Key, and Value
        # =====================

        # [sq, b, h] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = self.hidden_to_qkv(hidden_states, self.query_key_value)

        if self.is_knn() and self.memorize_mode == "train":
            mem_train = self.memory_train.get_partition(self.training)

            if not self.knn_use_pos_emb():
                hidden_states_to_mem = hidden_states.clone().detach()
                if self.attention_type == "knn_nopos"
                    query_layer_to_mem = query_layer.clone().detach()
        else:
            mem_train = None

        if exists(self.rotary_emb):
            if exists(self.rotary_ndims):
                raise BaseException("Not supported.")

                # partial rotary
                query_rot, query_pass = (
                    query_layer[..., : self.rotary_ndims],
                    query_layer[..., self.rotary_ndims :],
                )
                key_rot, key_pass = (
                    key_layer[..., : self.rotary_ndims],
                    key_layer[..., self.rotary_ndims :],
                )
            else:
                # full rotary
                query_rot, key_rot = query_layer, key_layer
            apply_rotary_fn = (
                apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb
            )

            seq_len = key_layer.shape[0]
            offset = 0

            if exists(mem_train) and self.attention_type != "knn_nopos":
				# Set aside positional embeddings for the memory
                offset += mem_train.get_pos_offset()

            if exists(layer_past) and layer_past.numel() > 0:
                offset += layer_past[0].shape[0]

            seq_len += offset

            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            query_layer, key_layer = apply_rotary_fn(
                query_rot, key_rot, cos, sin, offset=offset
            )

            if exists(self.rotary_ndims):
                query_layer = torch.cat((query_layer, query_pass), dim=-1)
                key_layer = torch.cat((key_layer, key_pass), dim=-1)

        if self.attention_type == "knn":
            raise ValueError("Not supported anymore.")

        # ==================================
        # Cache key and value for inference
        # ==================================

        if exists(layer_past) and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=0
            )

        if self.get_key_value:
            present = torch.stack((key_layer, value_layer))

        if not self.sparse:
            sz_past_keys = past_key.size(0) if exists(layer_past) and layer_past.numel() > 0 else 0
            sz_queries = query_layer.size(0)
            sz_keys = key_layer.size(0)

            # Shrink the attention mask to match the actual number of queries & keys, and exclude
            # past keys from queries
            attention_mask = attention_mask[:,:,sz_past_keys:sz_past_keys+sz_queries,:sz_keys]

            if exists(layer_past) and layer_past.numel() > 0:
                assert attention_mask.shape == (1, 1, 1, sz_keys)
                assert (~attention_mask).all()

            if not self.is_knn() or mem_train.is_empty():
                context_layer = self.attention(
                    query_layer, key_layer, value_layer, layer_past, attention_mask,
                )
            elif self.is_knn() and not mem_train.is_empty():
                if self.attention_type == "knn_both":
                    local_context_layer = self.attention(
                        query_layer, key_layer, value_layer, layer_past, attention_mask,
                    )

                    # Extract keys and values from memory
                    mem_keys, mem_vals, mem_mask = mem_train.get_memories(
                        key_layer.device,
                        self.training,
                        None,
                        eod_markers,
                        lambda context: self.hidden_to_qkv(context, self.query_key_value_mem)
                    )

                    mem_context_layer = self.attention(
                        query_layer,
                        mem_keys,
                        mem_vals,
                        None,
                        torch.cat(mem_mask))

                else:
                    # Extract keys and values from memory
                    mem_keys, mem_vals, mem_mask = mem_train.get_memories(
                        key_layer.device,
                        self.training,
                        query_layer_to_mem,
                        eod_markers,
                        lambda context: self.hidden_to_qkv(context, self.query_key_value)
                    )

                    if self.memory_attn_mode == "concat":
                        # Use a hybrid local-distant attention. Softmax will be applied across both
                        # local and distant logits, so that the attention head can attend locally
                        # as well as to the memorized entries.

                        # Add trained bias to the keys. The intent is to compensate for the lack of
                        # positional embedding.
                        if self.memory_key_bias != None:
                            mem_keys = mem_keys + self.memory_key_bias

                        # Concat memories with attention so that attention heads can attend to both
                        attention_mask = attention_mask.expand(mem_mask.shape[0], -1, -1, -1)
                        penalty_mem = (self.memory_penalty_factor * self.memory_penalty).expand(-1, -1, -1, mem_keys.shape[0])
                        penalty_zero = torch.full((1, penalty_mem.shape[1], 1, key_layer.shape[0]), 0, device=key_layer.device)

                        context_layer = self.attention(
                            query_layer,
                            torch.cat((mem_keys, key_layer)),
                            torch.cat((mem_vals, value_layer)),
                            None,
                            torch.cat((mem_mask, attention_mask), dim=3),
                            penalty=torch.cat((penalty_mem, penalty_zero), dim=3)
                        )
                    elif self.memory_attn_mode == "sigmoid":
                        # Use a sigmoid function to combine the local and distant attentions, like in
                        # https://arxiv.org/pdf/2203.08913.pdf.

                        mem_context_layer = self.attention(
                            query_layer,
                            mem_keys,
                            mem_vals,
                            None,
                            torch.cat(mem_mask))

                        local_context_layer = self.attention(
                            query_layer,
                            key_layer,
                            value_layer,
                            layer_past,
                            attention_mask)

                        # Use sigmoid to combine memories with attention
                        gate = (self.combine_attn_output_gate * 1000.0).sigmoid()
                        context_layer = local_context_layer * gate + mem_context_layer * (1 - gate)
                    else:
                        raise BaseException("Unknown memory_attn_mode", neox_args.memory_attn_mode)

        else:
            context_layer = self.sparse_attention(
                query_layer, key_layer, value_layer, attention_mask
            )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # Store the memories
        if self.is_knn():
            mem_train.add_memories(hidden_states_to_mem, eod_markers)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if self.get_key_value:
            output = [output, present]

        return output, bias

    def is_knn(self):
        """
        Whether distant KNN attention access is enabled
        """
        return self.attention_type == "knn"
            or self.attention_type == "knn_nopos"
            or self.attention_type == "knn_both"

    def knn_use_pos_emb(self):
        return not self.attention_type == "knn_nopos"

class ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        get_key_value=False,
    ):

        super().__init__()
        self.layer_number = layer_number

        norm, eps = get_norm(neox_args)

        # Layernorm on the input data.
        self.input_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.get_key_value = get_key_value

        self.hidden_dropout = neox_args.hidden_dropout
        self.bias_dropout_fusion = neox_args.bias_dropout_fusion
        self.gpt_j_residual = neox_args.gpt_j_residual

        if self.gpt_j_residual:
            self.reduce = mpu.mappings.reduce_from_model_parallel_region

        # Self attention.
        self.attention = ParallelSelfAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            get_key_value=self.get_key_value,
            rotary=rotary,
            parallel_output=self.gpt_j_residual,
        )

        # Layernorm on the output of the attention layer.
        self.post_attention_layernorm = norm(neox_args.hidden_size, eps=eps)

        # MLP
        self.mlp = ParallelMLP(
            neox_args=neox_args,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            parallel_output=self.gpt_j_residual,
        )

    def _get_bias_dropout(self):
        if self.bias_dropout_fusion:
            fn = (
                bias_dropout_add_fused_train
                if self.training
                else bias_dropout_add_fused_inference
            )
        else:
            fn = get_bias_dropout_add(self.training)
        return fn

    def forward(self, x, attention_mask, eod_markers, layer_past=None):
        bias_dropout_fn = self._get_bias_dropout()
        # x: [b, s, h]
        if self.gpt_j_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            # this means we can avoid doing the allreduce in the attn / mlp outputs
            # to save communication time (we can do a single allreduce after we add mlp / attn outputs).
            
            # attention_output = attn(ln1(x))
            residual = x
            attention_output, attention_bias = self.attention(
                self.input_layernorm(x), attention_mask, eod_markers, layer_past=layer_past
            )
            if self.get_key_value:
                attention_output, presents = attention_output

            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(attention_output),
                    residual=None,
                    prob=self.hidden_dropout,
                )

            # output = mlp(ln2(x)) + attention_output
            mlp_output, mlp_bias = self.mlp(self.post_attention_layernorm(x))
            with torch.enable_grad():
                output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(mlp_output),
                    residual=attention_output,
                    prob=self.hidden_dropout,
                )

            # output = output + residual
            output = residual + self.reduce(output)
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))

            residual = x

            # x = x + attn(ln1(x))
            attention_output, attention_bias = self.attention(
                self.input_layernorm(x), attention_mask, eod_markers, layer_past=layer_past
            )
            if self.get_key_value:
                attention_output, presents = attention_output
            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(residual),
                    residual=residual,
                    prob=self.hidden_dropout,
                )

            # output = x + mlp(ln2(x))
            mlp_output, mlp_bias = self.mlp(
                self.post_attention_layernorm(attention_output)
            )
            with torch.enable_grad():
                output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(attention_output),
                    residual=attention_output,
                    prob=self.hidden_dropout,
                )

        if self.get_key_value:
            output = [output, presents]

        return output


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline. """

    def forward(self, args):
        in_inference = len(args) == 5  # length of the args in inference == 5
        in_train = len(args) == 3  # length of the args in training == 3

        if in_train:
            hidden_states, eod_markers, attention_mask = args
            # we are returning just [hidden_states, mask]
            return super().forward(hidden_states, attention_mask, eod_markers), eod_markers, attention_mask
        elif in_inference:
            # we are in inference
            hidden_states, layer_past, presents, eod_markers, attention_mask = args

            past = torch.Tensor()
            if layer_past is not None and layer_past.numel() > 0:
                past = layer_past[self.layer_number]
            outputs = super().forward(hidden_states, attention_mask, eod_markers, layer_past=past)

            if self.get_key_value:
                # outputs = [hidden_states, present]
                hidden_states, present = outputs
                if presents.numel() == 0:
                    presents = present.unsqueeze(dim=0)
                else:
                    presents = torch.cat((presents, present.unsqueeze(dim=0)))
            else:
                hidden_states = outputs
            return hidden_states, layer_past, presents, eod_markers, attention_mask
        else:
            raise ValueError(
                f"In layer {self.layer_number} - Incorrect number of arguments ({len(args)}) for {self.__class__.__name__}"
            )


class ParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        if not isinstance(args, tuple):
            # in training, args = hidden_state (tensor, so we check if object isn't a tuple and pass through here)
            hidden_state = args
            logits, bias = super().forward(hidden_state)
            return logits
        elif len(args) == 2:
            # we are in inference, so input is (hidden_states, presents)
            hidden_state, presents = args
            logits, bias = super().forward(hidden_state)
            return logits, presents
        else:
            raise ValueError(
                f"Incorrect number of arguments for {self.__class__.__name__}"
            )


class NormPipe(nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, norm_class, hidden_size, eps):
        super().__init__()
        self.norm = norm_class(hidden_size, eps=eps)

    def forward(self, args):
        if not isinstance(args, tuple):
            # in training, args = hidden_state (tensor, so we check if object isn't a tuple and pass through here)
            hidden_state = args
            return self.norm(hidden_state)
        elif len(args) == 2:
            # in inference, args will be (hidden_state, presents)
            hidden_state, presents = args
            hidden_state = self.norm(hidden_state)
            return hidden_state, presents
        else:
            raise ValueError(
                f"Incorrect number of arguments for {self.__class__.__name__}"
            )


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_model_parallel_region(logits_parallel)
