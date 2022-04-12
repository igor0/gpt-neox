import torch

class SimpleMemory:
    def __init__(self, max_entries):
        self.max_entries = max_entries
        self.keys = None
        self.values = None
        self.eod_markers = None
        self.first_token = None

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
            # if there are any EOD markers, invalidate the memories up to (but excluding) the last marker
            if eod_markers[i][0] <= eod_markers[i][1]:
                self.valid_from[i] = self.keys.shape[0] - keys.shape[0] + eod_markers[i][1]
                self.first_token[i] = self.keys.shape[0] - keys.shape[0]

        # drop some memories if we already have too much

        if self.keys.shape[0] > self.max_entries:
            # shift the window forward
            removed_count = self.keys.shape[0] - self.max_entries
            self.keys = self.keys[removed_count:]
            self.values = self.values[removed_count:]

            for i in range(len(eod_markers)):
                self.valid_from[i] -= min(self.valid_from[i], removed_count)
                self.first_token[i] -= removed_count

    def get(self, query_count, eod_markers):
        # Mask away:
        #    - memorized keys from before EOS
        #    - queries from after EOS

        # memory_mask: [b, sq, sk]
        memory_mask = torch.full(
            size=(self.keys.shape[1], 1, query_count, self.keys.shape[0]),
            fill_value=True,
            device=self.keys.device)

        print("keys", len(self.keys), self.valid_from)
        for batch in range(memory_mask.shape[0]):
            keys_valid_from = self.valid_from[batch]
            queries_valid_to = eod_markers[batch][0]
            memory_mask[batch][:][:queries_valid_to][keys_valid_from:] = False
            memory_mask[batch][:][queries_valid_to:][:] = False
            #print(batch, "keys (", keys_valid_from, memory_mask.shape[3], ") queries (", 0, queries_valid_to, ")")

        return self.keys, self.values, memory_mask

    def is_empty(self):
        return self.keys is None

