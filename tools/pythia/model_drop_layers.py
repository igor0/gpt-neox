import os
import re
import sys


def get_layer_file_name_str(layer_id, model_id):
    return "layer_{}-model_{}-model_states.pt".format(layer_id, model_id)

def get_layer_file_name(layer_id, model_id):
    return get_layer_file_name_str("{:02d}".format(layer_id), "{:02d}".format(model_id))

class model_shrink:
    def __init__(self, orig_path, shrunk_path):
        self.orig_path = orig_path
        self.shrunk_path = shrunk_path

        extra_layers = 5 # layers other than the regular transformer layers
        self.orig_layers = self.get_orig_layers() - extra_layers

    def create_link(self, orig_layer_id, shrunk_layer_id, shrunk_path_variant):
        orig_name = get_layer_file_name(orig_layer_id, 0)
        shrunk_name = get_layer_file_name(shrunk_layer_id, 0)

        orig_layer_path = os.path.join(self.orig_path, orig_name)
        shrunk_layer_path = os.path.join(shrunk_path_variant, shrunk_name)

        #print("Linking {} to {}".format(orig_layer_path, shrunk_layer_path))
        os.symlink(orig_layer_path, shrunk_layer_path)

    def get_orig_layers(self):
        regex_str = get_layer_file_name_str(r'(\d\d)', r'(\d\d)')
        file_name_regex = re.compile(regex_str)

        orig_layers = 0
        for x in os.listdir(self.orig_path):
            print(x)
            m = file_name_regex.match(x)
            if m is not None:
                orig_layers = max(orig_layers, int(m[1]) + 1)

        return orig_layers

    def link_shrunk_model(self, shrunk_layers):
        shrunk_path_variant = os.path.join(self.shrunk_path, "shrunk{}".format(shrunk_layers))
        os.mkdir(shrunk_path_variant)
        self.create_link(0, 0, shrunk_path_variant)

        for i in range(shrunk_layers):
            self.create_link(i + 2, i + 2, shrunk_path_variant)

        self.create_link(self.orig_layers + 3, shrunk_layers + 3, shrunk_path_variant)
        self.create_link(self.orig_layers + 4, shrunk_layers + 4, shrunk_path_variant)

    def link(self):
        for i in range(self.orig_layers):
            self.link_shrunk_model(i + 1)

def main():
    orig_path = sys.argv[1]
    shrunk_path = sys.argv[2] if len(sys.argv) > 2 else orig_path
    m = model_shrink(orig_path, shrunk_path)
    m.link()

if __name__ == "__main__":
    main()
