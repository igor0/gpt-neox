import faiss
import pickle

from megatron.memorize.paths import (
    get_mem_header_path,
    get_mem_index_path,
    get_mem_keys_path,
    get_mem_values_path
)

class MemIndex:
    def __init__(self, header, index, keys, values):
        self.header = header
        self.index = index
        self.keys = keys
        self.values = values

    def search(self, queries):
        keys = self.index.search(queries)
        return keys

def load_memindex(path, layer_number):
    # Load the header
    header_file_path = get_mem_header_path(path, layer_number)
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

def build_memindex(path):
    for layer_number, attn in enumerate(neox_args.attention_config):
        if attn == "knn":
            _build_index(path, layer_number)

def _build_index(path, layer_number):
    """
    Loads tensors from a file and builds a faiss index.
    """
    mem_file_path = get_mem_dump_path(path, layer_number)
    header, keys, values = _load_kv_pairs(mem_file_path, layer_number)

    # Check dimensions of keys and values
    if keys.shape[1] != header['heads']:
        raise ValueError("The number of heads in the key tensor does not match the number of heads in the header.")
    if values.shape[1] != header['heads']:
        raise ValueError("The number of heads in the value tensor does not match the number of heads in the header.")
    if keys.shape[2] != header['dim']:
        raise ValueError("The dimension of the key tensor does not match the dimension in the header.")
    if values.shape[2] != header['dim']:
        raise ValueError("The dimension of the value tensor does not match the dimension in the header.")

    # head, sequence, dim
    keys = keys.view(keys.shape[0] * keys.shape[1], keys.shape[2])
    values = values.view(values.shape[0] * values.shape[1], values.shape[2])

    # Build the index
    print("Building index from {}".format(mem_file_path))
    index = faiss.IndexFlatL2(header["dim"])
    index.verbose = True
    index.add(keys)
    index.train()

    # Save the header
    header_file_path = get_mem_header_path(path, layer_number)
    with open(header_path, "wb") as f:
        pickle.dump(header, f)

    # Save the index
    index_file_path = get_mem_index_path(path, layer_number)
    print("Saving index to {}".format(index_file_path))
    faiss.write_index(index, index_file_path)

    # Save the keys
    keys_file_path = get_mem_keys_path(path, layer_number)
    print("Saving keys to {}".format(keys_file_path))
    np.save(keys_file_path, tensors)

    # Save the values
    values_file_path = get_mem_values_path(path, layer_number)
    print("Saving values to {}".format(values_file_path))
    np.save(values_file_path, tensors)

def _load_kv_pairs(input_path, layer_number):
    print("Loading memorized key-value pairs from {}".format(input_path))
    keys = []
    values = []
    with open(input_path, "rb") as f:
        header = pickle.load(f)

        while True:
            try:
                kv = pickle.load(f)
                keys.append(kv[0])
                values.append(kv[1])
            except EOFError:
                break

    return header, torch.cat(keys), torch.cat(values)
