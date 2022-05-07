import torch

class MemoryLive:
    """
    Represents a sliding window of memories tracked for a particular model.
    """

    def __init__(self, device, memory_size, memory_invalid_query_mode, training_partitions, memory_dumper_init=None):
        """
        memory_size:
            Number of key/value pairs (per batch & attention head) to retain

        memory_invalid_query_mode:
            How to populate the memory attention mask for queries that are past an EOD token in the
            current context and thus shouldn't have access to memories:
            
                * "first_token" attends to the latest token that was a first token in some context.
                  Due to causal attention, the first token in a context carries the least meaning.
                  Also, likely for related reasons, inactive attention heads are known to park
                  themselves on the first token. Empirically, this approach seems to work quite
                  well.

                * "all_tokens" attends to all tokens in memory uniformly

        memory_dumper:
            Optional MemoryDumper to persist any added memories.
        """
        self.device = device

        self.training = True

        if memory_dumper_init is None:
            memory_dumper_init = lambda training: None

        self.partitions = [
            _MemoryPartition(
                memory_size,
                memory_invalid_query_mode,
                memory_dumper_init(not self.training),
            )
        ] + [
            _MemoryPartition(
                memory_size,
                memory_invalid_query_mode,
                memory_dumper_init(self.training),
            ) for _ in range(training_partitions)
        ]
        self.idx = -1

    def get_partition(self, training, partition_idx):
        idx = 0
        if training:
            idx = partition_idx + 1

        # deactivate old partition
        if self.idx >= 0:
            self.partitions[self.idx]._move_to(torch.device('cpu'))

        if self.idx == 0:
            # Clear the evaluation memory at the end of each evaluation cycle
            self.partitions[self.idx]._clear()

        # activate the new partition
        self.partitions[idx]._move_to(self.device)

        # update the current index
        self.idx = idx

        # return the current partition
        return self.partitions[idx]


class _MemoryPartition:
    """
    Partition of key-value memories in a single transformer layer. We expect one partition
    for training and one for evaluation.
    """

    def __init__(self, memory_size, memory_invalid_query_mode, memory_dumper=None):
        self.memory_size = memory_size
        self.memory_invalid_query_mode = memory_invalid_query_mode
        self.memory_dumper = memory_dumper
        self._clear()

    def _move_to(self, device):
        if self.context is not None:
            self.context = self.context.to(device)

    def _clear(self):
        self.context = None
        self.first_token = None
        self.pos_offset = 0

    def _sync(self):
        if self.memory_dumper is not None:
            self.memory_dumper.sync()

    def add_memories(self, context, eod_markers):
        """
            context: [sq, b, h]
            eod_markers
        """

        # adjust the offset to apply to the positional embedding

        self.pos_offset += context.shape[0] * context.shape[1]

        # save the memories to the file, if requested

        #if self.memory_dumper is not None:
        #    self.memory_dumper.dump([
        #        keys.view(keys.shape[0] * keys.shape[1], keys.shape[2], keys.shape[3]).cpu(),
        #        values.view(keys.shape[0] * keys.shape[1], keys.shape[2], keys.shape[3]).cpu(),
        #    ])

        # record the memories

        if self.context is None:
            self.context = context
            self.valid_from = [0] * len(eod_markers)
            self.first_token = [0] * len(eod_markers)
        else:
            self.context = torch.cat((self.context, context), dim=0)

        # invalidate any memories before the newest EOD token

        for i in range(len(eod_markers)):
            # update the "first token"
            self.first_token[i] = self.context.shape[0] - context.shape[0]

            # if there are any EOD markers, invalidate the memories up to (but excluding) the last marker
            if eod_markers[i][0] <= eod_markers[i][1]:
                self.valid_from[i] = self.context.shape[0] - context.shape[0] + eod_markers[i][1]

        # drop some memories if we already have too much

        if self.context.shape[0] > self.memory_size:
            # shift the window forward
            removed_count = self.context.shape[0] - self.memory_size
            self.context = self.context[removed_count:]

            for i in range(len(eod_markers)):
                self.valid_from[i] -= min(self.valid_from[i], removed_count)
                self.first_token[i] -= removed_count

    def get_memories(self, device, queries, eod_markers, qkv_func):
        # Mask away:
        #    - memorized keys from before EOS
        #    - queries from after EOS

        _, keys, values = qkv_func(self.context)

        # memory_mask: [b, head (broadcast), sq, sk]
        memory_mask = torch.full(
            size=(keys.shape[1], 1, queries.shape[0], keys.shape[0]),
            fill_value=True,
            device=device)

        for batch in range(memory_mask.shape[0]):
            keys_valid_from = self.valid_from[batch]
            queries_valid_to = eod_markers[batch][0]
            memory_mask[batch,:,:queries_valid_to,keys_valid_from:] = False

            if self.memory_invalid_query_mode == "first_token":
                memory_mask[batch,:,queries_valid_to:,self.first_token[batch]] = False
            elif self.memory_invalid_query_mode == "all_tokens":
                memory_mask[batch,:,queries_valid_to:,:] = False
            else:
                raise BaseException("Invalid memory_invalid_query_mode value", self.memory_invalid_query_mode)

        return keys, values, memory_mask

    def is_empty(self):
        return self.context is None

    def get_pos_offset(self):
        return self.pos_offset
