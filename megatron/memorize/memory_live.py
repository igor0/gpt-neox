import numpy as np
import torch

class MemoryLive:
    """
    Represents a sliding window of memories tracked for a particular model.
    """

    def __init__(self, device, memory_size, device_batch_size, hidden_size, memory_invalid_query_mode, training_partitions):
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
        """
        self.device = device

        self.training = True

        self.partitions = [
            _MemoryPartition(
                memory_size,
                device_batch_size,
                hidden_size,
                memory_invalid_query_mode,
            )
        ] + [
            _MemoryPartition(
                memory_size,
                device_batch_size,
                hidden_size,
                memory_invalid_query_mode,
            ) for _ in range(training_partitions)
        ]
        self.idx = -1

    def get_partition(self, training, partition_idx):
        idx = 0
        if training:
            idx = partition_idx + 1

        # Swap out partitions if necessary
        if idx != self.idx:
            if self.idx == 0:
                # Clear the evaluation memory at the end of each evaluation cycle
                self.partitions[self.idx]._clear()

            # update the current index
            self.idx = idx

        # return the current partition
        return self.partitions[idx]

class _MemoryBuffer:
    def __init__(self, memory_size, device_batch_size, hidden_size):
        self.memory_size = memory_size
        self.device_batch_size = device_batch_size
        self.hidden_size = hidden_size

        self.memories = np.zeros(
            (memory_size, device_batch_size, hidden_size),
            dtype=np.float16,
        )
        self.clear()

    def clear(self):
        self._head = 0
        self._count = 0

    def add(self, new_memories):
        new_memories = new_memories.cpu()

        tail = (self._head + self._count) % self.memories.shape[0]
        add_end = tail + new_memories.shape[0]
        if add_end <= self.memories.shape[0]:
            self.memories[tail:add_end,:,:] = new_memories
        else:
            chunk1_size = self.memories.shape[0] - tail
            chunk2_size = new_memories.shape[0] - chunk1_size
            self.memories[tail:,:,:] = new_memories[:chunk1_size,:,:]
            self.memories[:chunk2_size:,:] = new_memories[chunk1_size:,:,:]

        self._count += new_memories.shape[0]

        if self._count > self.memories.shape[0]:
            removed_count = self._count - self.memories.shape[0]
            self._count -= removed_count
            self._head = (self._head + removed_count) % self.memories.shape[0]
        else:
            removed_count = 0

        return removed_count

    def get(self, device):
        tail_no_wrap = self._head + self._count
        if tail_no_wrap <= self.memories.shape[0]:
            chunk = self.memories[self._head:tail_no_wrap,:,:]
            return torch.from_numpy(chunk).to(device)

        chunk1 = self.memories[self._head:,:,:]
        chunk2 = self.memories[:tail_no_wrap - self.memories.shape[0],:,:]
        return torch.cat((
            torch.from_numpy(chunk1).to(device),
            torch.from_numpy(chunk2).to(device)
        ), dim=0)

    def count(self):
        return self._count

class _MemoryPartition:
    """
    Partition of key-value memories in a single transformer layer. We expect one partition
    for training and one for evaluation.
    """

    def __init__(self, memory_size, device_batch_size, hidden_size, memory_invalid_query_mode):
        self.buffer = _MemoryBuffer(memory_size, device_batch_size, hidden_size)
        self.memory_invalid_query_mode = memory_invalid_query_mode

        self.valid_from = [0] * device_batch_size
        self.first_token = [0] * device_batch_size

        self._clear()

    def _clear(self):
        self.buffer.clear()

        for i in range(len(self.valid_from)):
            self.valid_from[i] = 0
            self.first_token[i] = 0

    def add_memories(self, new_memories, eod_markers):
        """
            new_memories: [sq, b, h]
            eod_markers
        """

        # record the memories
        removed_count = self.buffer.add(new_memories)

        # invalidate any memories before the newest EOD token
        for i in range(len(eod_markers)):
            # update the "first token"
            self.first_token[i] = self.buffer.count() - new_memories.shape[0]

            # if there are any EOD markers, invalidate the memories up to (but excluding) the last marker
            if eod_markers[i][0] <= eod_markers[i][1]:
                self.valid_from[i] = self.first_token[i] + eod_markers[i][1]

        # drop some memories if we already have too much

        if removed_count > 0:
            # the window shifted forward
            for i in range(len(eod_markers)):
                self.valid_from[i] -= min(self.valid_from[i], removed_count)
                self.first_token[i] -= removed_count

    def get_memories(self, device, queries, eod_markers, qkv_func):
        # Get the keys and values
        _, keys, values = qkv_func(self.buffer.get(device))

        # Mask away:
        #    - memorized keys from before EOS
        #    - queries from after EOS

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
        return self.buffer.count() == 0
