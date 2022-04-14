import torch

class SimpleMemory:
    def __init__(self, memory_size, memory_invalid_query_mode):
        self.memory_size = memory_size
        self.keys = None
        self.values = None
        self.eod_markers = None
        self.first_token = None
        self.memory_invalid_query_mode = memory_invalid_query_mode

    def add(self, keys, values, eod_markers):
        """
            keys: [sq, b, np, hn]
            values: [sq, b, np, hn]
            eod_markers
        """

        # record the memories

        if self.keys is None:
            self.keys = keys
            self.values = values
            self.valid_from = [0] * len(eod_markers)
            self.first_token = [0] * len(eod_markers)
        else:
            self.keys = torch.cat((self.keys, keys), dim=0)
            self.values = torch.cat((self.values, values), dim=0)

        # invalidate any memories before the newest EOD token

        for i in range(len(eod_markers)):
            # update the "first token"
            self.first_token[i] = self.keys.shape[0] - keys.shape[0]

            # if there are any EOD markers, invalidate the memories up to (but excluding) the last marker
            if eod_markers[i][0] <= eod_markers[i][1]:
                self.valid_from[i] = self.keys.shape[0] - keys.shape[0] + eod_markers[i][1]

        # drop some memories if we already have too much

        if self.keys.shape[0] > self.memory_size:
            # shift the window forward
            removed_count = self.keys.shape[0] - self.memory_size
            self.keys = self.keys[removed_count:]
            self.values = self.values[removed_count:]

            for i in range(len(eod_markers)):
                self.valid_from[i] -= min(self.valid_from[i], removed_count)
                self.first_token[i] -= removed_count

    def get(self, query_count, eod_markers):
        # Mask away:
        #    - memorized keys from before EOS
        #    - queries from after EOS

        # memory_mask: [b, 1, sq, sk]
        memory_mask = torch.full(
            size=(self.keys.shape[1], 1, query_count, self.keys.shape[0]),
            fill_value=True,
            device=self.keys.device)

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

        return self.keys, self.values, memory_mask

    def is_empty(self):
        return self.keys is None

