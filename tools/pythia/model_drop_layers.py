import os
import re
import shutil
import sys
import yaml


def get_layer_file_name_str(layer_id, model_id):
    return "layer_{}-model_{}-model_states.pt".format(layer_id, model_id)

def get_layer_file_name(layer_id, model_id):
    return get_layer_file_name_str("{:02d}".format(layer_id), "{:02d}".format(model_id))

class model_shrink:
    def __init__(self, checkpoint_path, shrunk_path_base):
        self.checkpoint_path = checkpoint_path
        self.shrunk_path_base = shrunk_path_base

        extra_layers = 5 # layers other than the regular transformer layers
        self.orig_layers = self.get_orig_layers() - extra_layers

        self.checkpoint_name = os.path.basename(self.checkpoint_path.rstrip(os.sep))


    def create_link(self, orig_layer_id, shrunk_layer_id, shrunk_path_variant):
        orig_name = get_layer_file_name(orig_layer_id, 0)
        shrunk_name = get_layer_file_name(shrunk_layer_id, 0)

        orig_layer_path = os.path.join(self.checkpoint_path, orig_name)
        shrunk_layer_path = os.path.join(shrunk_path_variant, shrunk_name)

        self.fs_link(orig_layer_path, shrunk_layer_path)

    def get_orig_layers(self):
        regex_str = get_layer_file_name_str(r'(\d\d)', r'(\d\d)')
        file_name_regex = re.compile(regex_str)

        orig_layers = 0
        for x in os.listdir(self.checkpoint_path):
            print(x)
            m = file_name_regex.match(x)
            if m is not None:
                orig_layers = max(orig_layers, int(m[1]) + 1)

        return orig_layers

    def link_shrunk_model(self, shrunk_layers):
        configs_dir = "configs"
        config_file = "config.yml"

        # Figure out the paths

        shrunk_path_variant = "{}.{}".format(self.shrunk_path_base, shrunk_layers)
        shrunk_checkpoint = os.path.join(shrunk_path_variant, self.checkpoint_name)
        shrunk_configs_path = os.path.join(shrunk_path_variant, configs_dir)
        shrunk_config = os.path.join(shrunk_configs_path, config_file)
        orig_config = os.path.join(self.checkpoint_path, "..", configs_dir, config_file)

        # Create the directories

        self.fs_mkdir(shrunk_path_variant)
        self.fs_mkdir(shrunk_checkpoint)
        self.fs_mkdir(shrunk_configs_path)

        # Read, modify and write the config file

        with open(orig_config) as conf_file:
            conf = yaml.load(conf_file, Loader=yaml.FullLoader)

        conf['num-layers'] = shrunk_layers
        conf['load'] = shrunk_path_variant
        conf['save'] = shrunk_path_variant

        print("Writing to", shrunk_config)
        with open(shrunk_config, 'w') as f:
            yaml.dump(conf, f)

        # Copy the 'latest' file

        self.fs_copy(os.path.join(self.checkpoint_path, "..", "latest"), shrunk_path_variant)

        # Create checkpoint links

        self.create_link(0, 0, shrunk_checkpoint)

        for i in range(shrunk_layers):
            self.create_link(i + 2, i + 2, shrunk_checkpoint)

        self.create_link(self.orig_layers + 3, shrunk_layers + 3, shrunk_checkpoint)
        self.create_link(self.orig_layers + 4, shrunk_layers + 4, shrunk_checkpoint)

        ms_file = "mp_rank_00_model_states.pt"
        ms_path_orig = os.path.join(self.checkpoint_path, ms_file)
        ms_path_shrunk = os.path.join(shrunk_checkpoint, ms_file)
        self.fs_link(ms_path_orig, ms_path_shrunk)

    def fs_link(self, src, tgt):
        print("Linking '{}' to '{}'".format(src, tgt))
        os.symlink(src, tgt)

    def fs_mkdir(self, path):
        print("Creating directory '{}'".format(path))
        os.mkdir(path)

    def fs_copy(self, src, dst):
        print("Creating '{}' to '{}'".format(src, dst))
        shutil.copy(src, dst)

    def link_models(self):
        for layers in range(self.orig_layers + 1):
            self.link_shrunk_model(layers)

def main():
    checkpoint_path = sys.argv[1]
    shrunk_path_base = sys.argv[2]
    m = model_shrink(checkpoint_path, shrunk_path_base)
    m.link_models()

if __name__ == "__main__":
    main()
