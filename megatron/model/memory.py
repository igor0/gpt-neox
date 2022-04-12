import torch

class SimpleMemory:
    def __init__(self, max_entries):
        self.max_entries = max_entries
        self.keys = None
        self.values = None
        self.mask = None

    def add(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = torch.cat((self.keys, keys), dim=0)[-self.max_entries:]
            self.values = torch.cat((self.values, values), dim=0)[-self.max_entries:]

    def get(self):
        mask = None # TODO
        return self.keys, self.values, mask

    def is_empty(self):
        return self.keys is None

