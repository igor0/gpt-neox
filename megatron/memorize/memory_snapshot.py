import faiss
import math
import numpy as np
import pickle
import torch

from megatron.utils import print_rank_0
from megatron.memorize.paths import *

class _MemorySnapshot:
    def __init__(self, header, layer_number, all_indexes, all_keys, all_values):
        self.header = header
        self.layer_number = layer_number
        self.all_indexes = all_indexes
        self.all_keys = all_keys
        self.all_values = all_values
        self.k = 32 # XXX - make this a config parameter

        self.stats = {}

    def get_partition(self, training):
        # Unlike with live memory, memory snapshot has no separate partition for evaluation and
        # training.
        return self

    def add_memories(self, *args, **kwargs):
        return

    def get_memories(self, device, is_training, queries, eod_markers):
        """
        # queries: [sq, b, np, hn]
        """
        # every query is allowed to attend to any memory

        # [tk, b, np, hn]
        res_keys = []
        res_values = []

        for head in range(self.header["heads"]):
            qhead = queries[:,:,head,:]
            qhead = qhead.view(qhead.shape[0] * qhead.shape[1], qhead.shape[2])
            qhead = np.float32(qhead.cpu().numpy())

            distances, ids = self.all_indexes[head].search(qhead, k=self.k)
            for i in range(ids.shape[0]):
                best_id = ids[i,0] if ids[i,0] != -1 else 0
                if best_id not in self.stats:
                    self.stats[best_id] = 0

                self.stats[best_id] += 1

            if False:
                for i in range(ids.shape[0]):
                    for j in range(ids.shape[1]):
                        id_adj = ids[i,j]
                        id_adj = id_adj if id_adj != -1 else 0
                        print("ID", ids[i,j], "->", id_adj, "DIST", distances[i,j], np.dot(qhead[i,:], self.all_keys[head][id_adj,:]))

                    best = -1
                    best_dist = 0
                    for j in range(self.all_keys[head].shape[0]):
                        dist = np.dot(qhead[i,:], self.all_keys[head][j,:])
                        if best == -1 or dist > best_dist:
                            best = j
                            best_dist = dist
                    print("BEST", best, best_dist)


                # If no match, simply use the first key. TODO: explain why
                print("XXX", "FOUND IDS", np.count_nonzero(ids != -1), "/", ids.size)

            ids[ids == -1] = 0
            ids = np.expand_dims(ids.flatten(), axis=1)
            
            # Select the keys and values
            # [tk, b, hn]
            keys = torch.from_numpy(self.all_keys[head][ids,:]).to(device)
            values = torch.from_numpy(self.all_values[head][ids,:]).to(device)

            res_keys.append(keys.unsqueeze(dim=2))
            res_values.append(values.unsqueeze(dim=2))

        # mask: [b, np, sq, sk]
        mask = torch.full(
            size=(queries.shape[1], 1, queries.shape[0], self.k * queries.shape[0]),
            fill_value=False,
            device = device)

        return torch.cat(res_keys, dim=2), torch.cat(res_values, dim=2), mask

    def is_empty(self):
        return False

    def get_pos_offset(self):
        return (self.all_keys[0].shape[0] if len(self.all_keys) > 0 else 0)

    def __del__(self):
        sorted_stats = {k: v for k, v in sorted(self.stats.items(), key=lambda item: -item[1])}
        print("[", self.layer_number, "]", sorted_stats)

def load_memory_snapshot(path, layer_number):
    # Load the header
    header_file_path = get_mem_metadata_path(path, layer_number)
    with open(header_file_path, "rb") as f:
        header = pickle.load(f)

    all_indexes = []
    all_keys = []
    all_values = []

    for head in range(header["heads"]):
        # Load the index
        index_file_path = get_mem_index_path(path, layer_number, head)
        print("Loading index from {}".format(index_file_path))
        index = faiss.read_index(index_file_path)
        all_indexes.append(index)

        # Load the keys
        keys_file_path = get_mem_keys_path(path, layer_number, head)
        print("Loading keys from {}".format(keys_file_path))
        keys = np.load(keys_file_path)
        all_keys.append(keys)

        # Load the values
        values_file_path = get_mem_values_path(path, layer_number, head)
        print("Loading values from {}".format(values_file_path))
        values = np.load(values_file_path)
        all_values.append(values)

    return _MemorySnapshot(header, layer_number, all_indexes, all_keys, all_values)

def index_memory_snapshot(path, layer_number):
    """
    Loads memorized key-value records from a file and builds a KNN index.
    """

    # Load memorized records
    mem_file_path = get_mem_dump_path(path, layer_number)
    metadata, keys, values = _load_kv_pairs(mem_file_path, layer_number)

    # Print info about what we loaded
    print_rank_0("    Layer:", mem_file_path)
    print_rank_0("    Metadata:", metadata)
    print_rank_0("    Keys:", keys.shape)
    print_rank_0("    Values:", values.shape)

    # Function to assert dimensions of loaded records
    def check_dim(dim_name, dim_got):
        dim_exp = metadata[dim_name]
        if dim_got != dim_exp:
            raise ValueError(f'Invalid record. "{dim_name}" was {dim_got}, expected {dim_exp}')

    # Check dimensions of keys and values
    check_dim('records_count', keys.shape[0])
    check_dim('records_count', values.shape[0])
    check_dim('heads', keys.shape[1])
    check_dim('heads', values.shape[1])
    check_dim('dim', keys.shape[2])
    check_dim('dim', values.shape[2])

    # Save the metadata
    metadata_file_path = get_mem_metadata_path(path, layer_number)
    with open(metadata_file_path, "wb") as f:
        pickle.dump(metadata, f)


    for head in range(metadata["heads"]):
        # Save the keys
        keys_file_path = get_mem_keys_path(path, layer_number, head)
        print("Saving keys to {}".format(keys_file_path))
        np.save(keys_file_path, keys[:, head, :])

        # Save the values
        values_file_path = get_mem_values_path(path, layer_number, head)
        print("Saving values to {}".format(values_file_path))
        np.save(values_file_path, values[:, head, :])

        # Build the index

        print(f'Building index from "{mem_file_path}" for head {head}')

        # Create a faiss KNN index for the keys
        dim = metadata["dim"]
        #quantizer = faiss.IndexFlatIP(dim)
        #nlist = min(4096, round(math.sqrt(keys.shape[0])))
        #index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexFlatIP(dim)

        # Convert the keys to float32 numpy array, which is what faiss expects
        keys_for_index = keys[:, head, :].numpy()
        if keys.dtype != np.float32:
            print(f'Converting keys from {keys.dtype} to float32')
            keys_for_index = np.float32(keys_for_index)

        # Train and populate the index
        index.verbose = True
        index.train(keys_for_index)
        index.add(keys_for_index)

        # Save the index
        index_file_path = get_mem_index_path(path, layer_number, head)
        print("Saving index to {}".format(index_file_path))
        faiss.write_index(index, index_file_path)

def _load_kv_pairs(input_path, layer_number):
    """
    Loads memorized key-value records from a file.
    """

    print("Loading memorized key-value pairs from {}".format(input_path))
    keys = []
    values = []
    with open(input_path, "rb") as f:
        header = pickle.load(f)

        while True:
            kv = pickle.load(f)
            if kv is None:
                break

            keys.append(kv[0])
            values.append(kv[1])

        footer = pickle.load(f)

    # Merge the header and footer records
    metadata = {**header, **footer}

    return metadata, torch.cat(keys), torch.cat(values)
