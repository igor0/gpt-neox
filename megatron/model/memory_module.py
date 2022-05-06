import math
import torch
import torch.nn as nn

from .norms import get_norm
from megatron import mpu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax

class ParallelMemoryModule(nn.Module):
    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
    ):

        super().__init__()
        self.layer_number = layer_number

        norm, eps = get_norm(neox_args)

        # Layer Norms
        self.layernorm1 = norm(neox_args.hidden_size, eps=eps)
        self.layernorm2 = norm(neox_args.hidden_size, eps=eps)
        self.layernorm3 = norm(neox_args.hidden_size, eps=eps)

        # MLP 1
        self.mlp1 = ParallelMLP(
            neox_args=neox_args,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            parallel_output=False,
        )

        # Self attention.
        self.attn = MemoryAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            parallel_output=False,
        )

        # MLP 2
        self.mlp2 = ParallelMLP(
            neox_args=neox_args,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            parallel_output=False,
        )

        # Validate arguments
        if neox_args.hidden_dropout != 0:
            raise ValueError(f'ParallelMemoryModule does not support hidden_dropout (hidden_dropout={neox_args.hidden_dropout})')

    def forward(self, x):
        # pseudocode:
        #
        # x = mlp(ln1(x))
        # x = x + attn(ln2(x))
        # x = x + mlp(ln3(x))

        # x = mlp1(ln1(x))
        mlp1_output, mlp1_bias = self.mlp1(self.layernorm1(x))
        x = mlp1_output + mlp1_bias.expand_as(mlp1_output)

        # x = x + attn(ln2(x))
        attention_output, attention_bias = self.attn(self.layernorm2(x))
        x = x + attention_output + attention_bias.expand_as(residual)

        # x = x + mlp2(ln3(x))
        mlp2_output, mlp2_bias = self.mlp2(self.layernorm3(x))
        x = x + mlp2_output + mlp2_bias.expand_as(mlp2_output)

        return x

class MemoryAttention(nn.Module):
    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        parallel_output=False,
    ):
        super().__init__()

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
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
        self.query = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            gather_output=False,
            init_method=init_method,
        )

        self.key_value = mpu.ColumnParallelLinear(
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

    def attention(
        self, query_layer, key_layer, value_layer, attention_mask,
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

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

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

    def hidden_to_kv(self, hidden_states, kv, normalize):
        # Attention heads [sq, b, h] --> [sq, b, (np * 2 * hn)]
        mixed_x_layer, _ = kv(hidden_states)

        # [sq, b, (np * 2 * hn)] --> [sq, b, np, 2 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 2 * hn] --> 2 [sq, b, np, hn]
        (key_layer, value_layer) = mpu.split_tensor_along_last_dim(
            mixed_x_layer, 2
        )
        if normalize:
            key_layer = l2norm(key_layer)
            value_layer = l2norm(value_layer)
        return key_layer, value_layer

    def forward(self, hidden_states, eod_markers):
        # hidden_states: [sq, b, h]

        if self.memorize_mode == "train":
            mem_train = self.memory_train.get_partition(self.training)

            if self.attention_type != "knn_both":
                raise ValueError("Not supported.")
        else:
            mem_train = None

        sz_queries = query_layer.size(0)
        sz_keys = key_layer.size(0)

        if mem_train is None or mem_train.is_empty():
            output, bias = None, None
        else:
            # [sq, b, h] --> [sq, b, np * hn]
            query, _ = self.query(hidden_states)
            query = l2norm(query)

            # Extract keys and values from memory
            mem_keys, mem_vals, mem_mask = mem_train.get_memories(
                key_layer.device,
                self.training,
                query,
                eod_markers,
                lambda past_hidden_states: self.hidden_to_kv(past_hidden_states, normalize=True)
            )

            context_layer = self.attention(
                query,
                mem_keys,
                mem_vals,
                mem_mask,
            )

            output, bias = self.densify(context_layer, self.dense)

        # Store the memories
        mem_train.add_memories(hidden_states.clone().detach(), eod_markers)

        return output, bias

    def densify(self, context_layer, dense):
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)
       # =================
        # Output. [sq, b, h]
        # =================

        output, bias = dense(context_layer)
        return output, bias
