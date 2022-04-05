import os
import math
import torch
import faiss
import numpy as np

from einops import rearrange
from memorizing_transformers_pytorch.utils import rearrange_with_dim_list

# constants

FAISS_INDEX_GPU_ID = int(os.getenv('FAISS_INDEX_GPU_ID', 0))

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_list(val):
    return val if isinstance(val, list) else [val]

def check_shape(t, pattern, **kwargs):
    return rearrange(t, f"{pattern} -> {pattern}", **kwargs)

def count_intersect(x, y):
    # returns an array that shows how many times an element in x is contained in tensor y
    return np.sum(rearrange(x, 'i -> i 1') == rearrange(y, 'j -> 1 j'), axis = -1)

class KNN():
    def __init__(
        self,
        dim,
        max_num_entries,
        M = 15,
        use_gpu = False,
        cap_num_entries = False,
        flat = False
    ):
        if flat:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.use_gpu = use_gpu

        self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), FAISS_INDEX_GPU_ID, index) if use_gpu else index
        self.max_num_entries = max_num_entries
        self.cap_num_entries = cap_num_entries

        self.reset()

    def __del__(self):
        if hasattr(self, 'index'):
            del self.index

    def reset(self):
        self.index.reset()
        self.num_entries = 0

    def add(self, x):
        if self.cap_num_entries and self.num_entries > self.max_num_entries:
            self.reset()

        self.num_entries += x.shape[0]
        return self.index.add(x)

    def search(
        self,
        x,
        topk,
    ):
        distances, indices = self.index.search(x, k = topk)
        return indices

class KNNMemory():
    def __init__(
        self,
        dim,
        init_db,
        max_memories = 16000,
        num_indices = 1,
        knn_use_gpu = False,
        knn_flat = True
    ):
        self.dim = dim
        self.num_indices = num_indices
        self.max_memories = max_memories
        self.shape = (num_indices, max_memories, 2, dim)
        self.db_offsets = np.zeros(num_indices, dtype = np.int32)

        self.db = init_db(self.shape)
        self.knns = [KNN(dim = dim, max_num_entries = max_memories, use_gpu = knn_use_gpu, cap_num_entries = True, flat = knn_flat) for _ in range(num_indices)]

    def clear(self, indices = None):
        if not exists(indices):
            indices = list(range(self.num_indices))

        indices = cast_list(indices)

        for index in indices:
            knn = self.knns[index]
            knn.reset()

        self.db_offsets[indices] = 0

    def add(self, memories):
        check_shape(memories, 'b n kv d', d = self.dim, kv = 2, b = self.num_indices)

        memories = memories.detach().cpu().numpy()
        memories = memories[:, -self.max_memories:]
        num_memories = memories.shape[1]

        keys = np.ascontiguousarray(memories[..., 0, :])

        for key, knn in zip(keys, self.knns):
            knn.add(key)

        add_indices = (rearrange(np.arange(num_memories), 'j -> 1 j') + rearrange(self.db_offsets, 'i -> i 1')) % self.max_memories
        self.db[rearrange(np.arange(self.num_indices), 'i -> i 1'), add_indices] = memories
        self.db.flush()

        self.db_offsets += num_memories

    def search(
        self,
        queries,
        topk
    ):
        _, *prec_dims, _ = queries.shape
        check_shape(queries, 'b ... d', d = self.dim, b = self.num_indices)
        queries = rearrange(queries, 'b ... d -> b (...) d')

        device = queries.device
        queries = queries.detach().cpu().numpy()

        all_masks = []
        all_key_values = []

        for ind, (query, knn) in enumerate(zip(queries, self.knns)):
            indices = knn.search(query, topk)
            mask = indices !=  -1
            db_indices = np.where(mask, indices, 0)

            all_masks.append(torch.from_numpy(mask))

            key_values = self.db[ind, db_indices % self.max_memories]
            all_key_values.append(torch.from_numpy(key_values))

        all_masks = torch.stack(all_masks)
        all_key_values = torch.stack(all_key_values)
        all_key_values = all_key_values.masked_fill(~rearrange(all_masks, '... -> ... 1 1'), 0.)

        all_key_values = rearrange_with_dim_list(all_key_values, 'b (...p) ... -> b ...p ...', p = prec_dims)
        all_masks = rearrange_with_dim_list(all_masks, 'b (...p) ... -> b ...p ...', p = prec_dims)

        return all_key_values.to(device), all_masks.to(device)

    def __del__(self):
        if hasattr(self, 'knns'):
            for knn in self.knns:
                del knn
        del self.db
