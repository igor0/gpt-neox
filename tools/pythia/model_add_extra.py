import os
import re
import sys
import torch


def get_layer_file_name_str(layer_id, model_id):
    return "layer_{}-model_{}-model_states.pt".format(layer_id, model_id)

def get_layer_file_name(layer_id, model_id):
    return get_layer_file_name_str("{:02d}".format(layer_id), "{:02d}".format(model_id))


class checkpoint_info:
    def __init__(self, path, model_id):
        self.path = path
        self.model_id = model_id
        self.inspect_checkpoint()

    def inspect_checkpoint(self):
        file_name_regex = re.compile(get_layer_file_name_str(r'(\d\d)', r'(\d\d)'))

        max_layer_id = -1

        for x in os.listdir(self.path):
            m = file_name_regex.match(x)
            if m is not None:
                model_id = int(m[2])
                layer_id = int(m[1])

                if model_id == self.model_id:
                    max_layer_id = max(max_layer_id, layer_id)

        self.max_layer_id = max_layer_id

    def modify_layer_checkpoint(self):
        name = get_layer_file_name(self.max_layer_id, self.model_id)
        m = torch.load(os.path.join(self.path, name))

        final_linear = m["final_linear.weight"]
        dim = final_linear.shape[1]

        del m["final_linear.weight"]
        m["extra_linear.weight"] = torch.eye(dim).float()
        m["final_linear.weight"] = final_linear

        for k in m:
            print("{}: {} {}".format(k, m[k].shape, m[k].size()))

        print("Saving {}".format(name))
        os.rename(os.path.join(self.path, name), os.path.join(self.path, name + ".orig"))
        torch.save(m, os.path.join(self.path, name))
        return dim

    def modify_optimizer_checkpoint(self, dim):
        name = "mp_rank_00_model_states.pt"
        m = torch.load(os.path.join(self.path, name))
        m["optimizer"]["fp32_groups_flat"] = [torch.zeros(dim * dim, dtype=torch.float32)]

        print("Saving {}".format(name))
        os.rename(os.path.join(self.path, name), os.path.join(self.path, name + ".orig"))
        torch.save(m, os.path.join(self.path, name))

def main():
    orig_path = sys.argv[1]

    checkpoint = checkpoint_info(orig_path, 0)
    dim = checkpoint.modify_layer_checkpoint()
    checkpoint.modify_optimizer_checkpoint(dim)

if __name__ == "__main__":
    main()
