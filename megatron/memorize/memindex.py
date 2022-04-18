import faiss
import math
import numpy as np
import pickle
import torch

from megatron.utils import print_rank_0

from megatron.memorize.paths import (
    get_mem_metadata_path,
    get_mem_index_path,
    get_mem_keys_path,
    get_mem_values_path,
    get_mem_dump_path,
)

class MemIndex:
    def __init__(self, header, index, keys, values):
        self.header = header
        self.index = index
        self.keys = keys
        self.values = values

    def add_memories(self, *args, *kwargs):
        return

    def get_memories(self, is_training, query_count, eod_markers):
        # TODO
        queries = None
        keys = self.index.search(queries)

    def is_empty(self):
        return False

def load_memindex(path, layer_number):
    # Load the header
    header_file_path = get_mem_metadata_path(path, layer_number)
    with open(header_file_path, "rb") as f:
        header = pickle.load(f)

    # Load the index
    index_file_path = get_mem_index_path(path, layer_number)
    print("Loading index from {}".format(index_file_path))
    index = faiss.read_index(index_file_path)

    # Load the keys
    keys_file_path = get_mem_keys_path(path, layer_number)
    print("Loading keys from {}".format(keys_file_path))
    keys = np.load(keys_file_path)

    # Load the values
    values_file_path = get_mem_values_path(path, layer_number)
    print("Loading values from {}".format(values_file_path))
    values = np.load(values_file_path)

    return MemIndex(header, index, keys, values)

def build_memindex(path, attention_config):
    """
    Builds memindex for all memorizing layers in the model.
    """

    for layer_number, attn in enumerate(attention_config):
        if attn == "knn":
            _build_index(path, layer_number)

def _build_index(path, layer_number):
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

    # Reshape the keys and values into a list of embeddings
    # [head, sequence, dim] -> [head * sequence, dim]
    keys = keys.view(keys.shape[0] * keys.shape[1], keys.shape[2])
    values = values.view(values.shape[0] * values.shape[1], values.shape[2])

    # Save the metadata
    metadata_file_path = get_mem_metadata_path(path, layer_number)
    with open(metadata_file_path, "wb") as f:
        pickle.dump(metadata, f)

    # Save the keys
    keys_file_path = get_mem_keys_path(path, layer_number)
    print("Saving keys to {}".format(keys_file_path))
    np.save(keys_file_path, keys)

    # Save the values
    values_file_path = get_mem_values_path(path, layer_number)
    print("Saving values to {}".format(values_file_path))
    np.save(values_file_path, values)

    # Build the index

    print(f'Building index from "{mem_file_path}"')

    # Create a faiss KNN index for the keys
    dim = metadata["dim"]
    quantizer = faiss.IndexFlatIP(dim)
    nlist = min(4096, round(math.sqrt(keys.shape[0])))
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # Convert the keys to float32 numpy array, which is what faiss expects
    keys_for_index = keys.numpy()
    if keys.dtype != np.float32:
        print(f'Converting keys from {keys.dtype} to float32')
        keys_for_index = np.float32(keys_for_index)

    # Train and populate the index
    index.verbose = True
    index.train(keys_for_index)
    index.add(keys_for_index)

    # Save the index
    index_file_path = get_mem_index_path(path, layer_number)
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
