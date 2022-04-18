import os

def get_mem_dump_path(dir_path, layer_number):
    return os.path.join(dir_path, f'layer.{layer_number}.pkl')

def get_mem_header_path(dir_path, layer_number):
    return os.path.join(dir_path, f'layer.{layer_number}.header.json')

def get_mem_index_path(dir_path, layer_number):
    return os.path.join(dir_path, f'layer.{layer_number}.index')

def get_mem_keys_path(dir_path, layer_number):
    return os.path.join(dir_path, f'layer.{layer_number}.keys.npy')

def get_mem_values_path(dir_path, layer_number):
    return os.path.join(dir_path, f'layer.{layer_number}.values.npy')
