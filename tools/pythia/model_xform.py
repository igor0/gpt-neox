import argparse
import os
import re
import sys
import torch
import yaml

def fs_link(src, dst):
    print("Linking '{}' <- '{}'".format(src, dst))
    os.symlink(src, dst)

def fs_mkdir(path):
    print("Creating directory '{}'".format(path))
    os.mkdir(path)

def fs_copy(src, dst):
    print("Creating '{}' to '{}'".format(src, dst))
    shutil.copy(src, dst)

def fs_rename(src, dst):
    print("Renaming '{}' to '{}'".format(src, dst))
    os.rename(src, dst)

def get_layer_file_name_str(layer_id, model_id):
    return "layer_{}-model_{}-model_states.pt".format(layer_id, model_id)

def get_layer_file_name(layer_id, model_id):
    return get_layer_file_name_str("{:02d}".format(layer_id), "{:02d}".format(model_id))

def enable_extra_linear(args):
    return args.mode == 'extra_linear' or args.mode == 'out_linear_all'

class model:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model_id = 0
        self._inspect_checkpoint()

    def _inspect_checkpoint(self):
        file_name_regex = re.compile(get_layer_file_name_str(r'(\d\d)', r'(\d\d)'))

        max_layer_id = -1

        for x in os.listdir(self.checkpoint_path):
            m = file_name_regex.match(x)
            if m is not None:
                model_id = int(m[2])
                layer_id = int(m[1])

                if model_id == self.model_id:
                    max_layer_id = max(max_layer_id, layer_id)

        self.max_layer_id = max_layer_id

        extra_layers = 5 # layers other than the regular transformer layers
        self.layers_num = (self.max_layer_id + 1) - extra_layers

class model_transform:
    def __init__(self, args):
        checkpoint_path = os.path.join(args.orig_model_path, args.orig_checkpoint)

        self.args = args
        self.orig = model(checkpoint_path)

    def link_new_model(self, new_layers_num):
        configs_dir = "configs"
        config_file = "config.yml"
        baseline_checkpoint_name = "baseline"

        # Figure out the paths

        new_checkpoint_path = os.path.join(self.args.new_model_path, baseline_checkpoint_name)
        new_configs_path = os.path.join(self.args.new_model_path, configs_dir)
        new_config_path = os.path.join(new_configs_path, config_file)
        orig_config_path = os.path.join(self.args.orig_model_path, configs_dir, config_file)

        # Create the directories

        fs_mkdir(self.args.new_model_path)
        fs_mkdir(new_checkpoint_path)
        fs_mkdir(new_configs_path)

        # Read, modify and write the config file

        with open(orig_config_path) as conf_file:
            conf = yaml.load(conf_file, Loader=yaml.FullLoader)

        conf['load'] = self.args.new_model_path
        conf['save'] = self.args.new_model_path
        conf['finetune'] = True
        conf['num-layers'] = new_layers_num

        # Other args that we need - may need to be reconfigured
        if self.args.mode == 'final_norm':
            # The final layer norm is in its own pre-final "layer"
            #
            # Example of a 1-layer model:
            #   0.word_embeddings.weight False None
            #   2.{input_layernorm,attention,post_attention,mlp}
            #   4.norm
            #   5.final_linear
            conf['pythia_train_only'] = r'^{}\.norm'.format(self.max_layer_id - 1)
        elif self.args.mode == 'out_linear_all':
            conf['pythia_train_only'] = r'attention\.dense|dense_4h_to_h|extra_linear'
        elif self.args.mode == 'all' or self.args.mode == 'all_100k':
            del conf['pythia_train_only']
        else:
            conf['pythia_train_only'] = self.args.mode

        if enable_extra_linear(self.args):
            conf['pythia_extra_linear'] =  True

        if self.args.mode == 'all_100k':
            train_iters = 100_000
        elif self.args.predict == 'self':
            train_iters = 2_000
        else:
            train_iters = 10_000

        conf['train-iters'] = train_iters
        conf['lr-decay-iters'] = train_iters
        conf['data-path'] = '/mnt/ssd-1/data/pile_00/pile_00_text_document'
        conf['zero_optimization']['stage'] = 0

        if self.args.mode == 'final_linear':
            # Ultimately, different LRs seem to work for different depths here.
            # For dense_small, I believe I used 6e-4 for depths 0..7 and 6e-5
            # for depths 8..12.
            conf['optimizer']['params']['lr'] = 0.00006

        if self.args.predict is not None:
            conf['pythia_predict_special'] = self.args.predict
            #if self.args.predict == 'sink':
                #conf['optimizer']['params']['lr'] = 0.00006

        print("Writing to", new_config_path)
        with open(new_config_path, 'w') as f:
            yaml.dump(conf, f)

        # Create the 'latest' file

        new_latest_file_path = os.path.join(self.args.new_model_path, "latest")
        print("Writing to", new_latest_file_path)
        with open(new_latest_file_path, 'w') as fh:
            fh.write(baseline_checkpoint_name)

        # Create checkpoint links

        self.create_link(self.orig, 0, 0, new_checkpoint_path)

        for i in range(new_layers_num):
            self.create_link(self.orig, i + 2, i + 2, new_checkpoint_path)

        # Link the normalization layer
        self.create_link(self.orig, self.orig.layers_num + 3, new_layers_num + 3, new_checkpoint_path)

        # Link the final head layer
        model_head = self.orig if self.args.head is None else model(self.args.head)
        self.create_link(model_head, model_head.layers_num + 4, new_layers_num + 4, new_checkpoint_path)

        # Link the model states
        ms_file = "mp_rank_00_model_states.pt"
        orig_ms_path = os.path.join(self.orig.checkpoint_path, ms_file)
        new_ms_path = os.path.join(new_checkpoint_path, ms_file)
        fs_link(orig_ms_path, new_ms_path)

        return mutable_model(self.args, new_checkpoint_path, new_layers_num, new_layers_num + 4, self.orig.model_id)

    @staticmethod
    def create_link(orig, orig_layer_id, new_layer_id, new_checkpoint_path):
        orig_name = get_layer_file_name(orig_layer_id, 0)
        new_name = get_layer_file_name(new_layer_id, 0)

        orig_layer_path = os.path.join(orig.checkpoint_path, orig_name)
        new_layer_path = os.path.join(new_checkpoint_path, new_name)

        fs_link(orig_layer_path, new_layer_path)


class mutable_model:
    def __init__(self, args, checkpoint_path, new_layers_num, max_layer_id, model_id):
        self.args = args
        self.checkpoint_path = checkpoint_path
        self.new_layers_num = new_layers_num
        self.max_layer_id = max_layer_id
        self.model_id = model_id

    def modify_layer_checkpoint(self):
        name = get_layer_file_name(self.max_layer_id, self.model_id)
        layer_chkpt_path = os.path.join(self.checkpoint_path, name)
        m = torch.load(layer_chkpt_path, map_location=torch.device('cpu'))

        final_linear = m["final_linear.weight"]
        dim = final_linear.shape[1]

        # If there is a newly added extra_linear head, initialize it to identity
        if enable_extra_linear(self.args) and self.args.head is None:
            del m["final_linear.weight"]
            m["extra_linear.weight"] = torch.eye(dim).float()
            m["final_linear.weight"] = final_linear

            # Rename for 2 reasons: keep the backup copy & also if symlink, don't overwrite the destination file
            fs_rename(layer_chkpt_path, layer_chkpt_path + ".orig")
            print("Saving {}".format(layer_chkpt_path))

            # Save the updated checkpoint
            torch.save(m, layer_chkpt_path)

        del m
        return dim

    def modify_optimizer_checkpoint(self, dim):
        name = "mp_rank_00_model_states.pt"
        layer_chkpt_path = os.path.join(self.checkpoint_path, name)
        checkpoint = torch.load(layer_chkpt_path, map_location=torch.device('cpu'))
        #checkpoint["optimizer"]["fp32_groups_flat"] = []
        if 'args' in checkpoint and 'num_layers' in checkpoint['args']:
            checkpoint['args']['num_layers'] = self.new_layers_num

        # Rename for 2 reasons: keep the backup copy & also if symlink, don't overwrite the destination file
        fs_rename(layer_chkpt_path, layer_chkpt_path + ".orig")

        # Save the updated checkpoint
        print("Saving {}".format(name))
        torch.save(checkpoint, layer_chkpt_path)
        del checkpoint

def canonicalize_args(args):
    args.orig_model_path = os.path.abspath(args.orig_model_path)
    args.new_model_path = os.path.abspath(args.new_model_path)
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="extra_linear", choices=['extra_linear', 'final_linear', 'final_norm', 'out_linear_all', 'all', 'all_100k'])
    parser.add_argument("--head", type=str)
    parser.add_argument("--predict", type=str, choices=['self', 'abs', 'prev', 'sink'])
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("orig_model_path")
    parser.add_argument("orig_checkpoint")
    parser.add_argument("new_model_path")

    args = canonicalize_args(parser.parse_args())

    transform = model_transform(args)

    layers_num = args.num_layers if args.num_layers is not None else transform.orig.layers_num
    mutable = transform.link_new_model(layers_num)
    dim = mutable.modify_layer_checkpoint()
    mutable.modify_optimizer_checkpoint(dim)

if __name__ == "__main__":
    main()
