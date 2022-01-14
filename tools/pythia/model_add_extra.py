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


class model_transform:
    def __init__(self, orig_model_path, orig_checkpoint, new_model_path, model_id):
        self.orig_model_path = orig_model_path
        self.orig_checkpoint = orig_checkpoint
        self.orig_checkpoint_path = os.path.join(orig_model_path, orig_checkpoint)
        self.new_model_path = new_model_path
        self.model_id = model_id

        self.inspect_checkpoint()

    def inspect_checkpoint(self):
        file_name_regex = re.compile(get_layer_file_name_str(r'(\d\d)', r'(\d\d)'))

        max_layer_id = -1

        for x in os.listdir(self.orig_checkpoint_path):
            m = file_name_regex.match(x)
            if m is not None:
                model_id = int(m[2])
                layer_id = int(m[1])

                if model_id == self.model_id:
                    max_layer_id = max(max_layer_id, layer_id)

        self.max_layer_id = max_layer_id

        extra_layers = 5 # layers other than the regular transformer layers
        self.orig_layers_num = (self.max_layer_id + 1) - extra_layers

    def link_new_model(self, new_layers_num):
        configs_dir = "configs"
        config_file = "config.yml"
        baseline_checkpoint_name = "baseline"

        # Figure out the paths

        new_checkpoint_path = os.path.join(self.new_model_path, baseline_checkpoint_name)
        new_configs_path = os.path.join(self.new_model_path, configs_dir)
        new_config_path = os.path.join(new_configs_path, config_file)
        orig_config_path = os.path.join(self.orig_model_path, configs_dir, config_file)

        # Create the directories

        fs_mkdir(self.new_model_path)
        fs_mkdir(new_checkpoint_path)
        fs_mkdir(new_configs_path)

        # Read, modify and write the config file

        with open(orig_config_path) as conf_file:
            conf = yaml.load(conf_file, Loader=yaml.FullLoader)

        conf['load'] = self.new_model_path
        conf['save'] = self.new_model_path
        conf['finetune'] = True

        # Other args that we need - may need to be reconfigured
        conf['pythia_train_only'] = 'extra_linear'
        conf['pythia_extra_linear'] =  True
        conf['train-iters'] = 10000
        conf['lr-decay-iters'] = 10000
        conf['data-path'] = '/mnt/ssd-1/data/pile_00/pile_00_text_document'
        conf['zero_optimization']['stage'] = 0

        print("Writing to", new_config_path)
        with open(new_config_path, 'w') as f:
            yaml.dump(conf, f)

        # Create the 'latest' file

        new_latest_file_path = os.path.join(self.new_model_path, "latest")
        print("Writing to", new_latest_file_path)
        with open(new_latest_file_path, 'w') as fh:
            fh.write(baseline_checkpoint_name)

        # Create checkpoint links

        self.create_link(0, 0, new_checkpoint_path)

        for i in range(new_layers_num):
            self.create_link(i + 2, i + 2, new_checkpoint_path)

        self.create_link(self.orig_layers_num + 3, new_layers_num + 3, new_checkpoint_path)
        self.create_link(self.orig_layers_num + 4, new_layers_num + 4, new_checkpoint_path)

        ms_file = "mp_rank_00_model_states.pt"
        orig_ms_path = os.path.join(self.orig_checkpoint_path, ms_file)
        new_ms_path = os.path.join(new_checkpoint_path, ms_file)
        fs_link(orig_ms_path, new_ms_path)


        return mutable_model(new_checkpoint_path, new_layers_num, self.max_layer_id, self.model_id)

    def create_link(self, orig_layer_id, new_layer_id, new_checkpoint_path):
        orig_name = get_layer_file_name(orig_layer_id, 0)
        new_name = get_layer_file_name(new_layer_id, 0)

        orig_layer_path = os.path.join(self.orig_checkpoint_path, orig_name)
        new_layer_path = os.path.join(new_checkpoint_path, new_name)

        fs_link(orig_layer_path, new_layer_path)


class mutable_model:
    def __init__(self, checkpoint_path, new_layers_num, max_layer_id, model_id):
        self.checkpoint_path = checkpoint_path
        self.new_layers_num = new_layers_num
        self.max_layer_id = max_layer_id
        self.model_id = model_id

    def modify_layer_checkpoint(self):
        name = get_layer_file_name(self.max_layer_id, self.model_id)
        layer_chkpt_path = os.path.join(self.checkpoint_path, name)
        m = torch.load(layer_chkpt_path)

        final_linear = m["final_linear.weight"]
        dim = final_linear.shape[1]

        del m["final_linear.weight"]
        m["extra_linear.weight"] = torch.eye(dim).float()
        m["final_linear.weight"] = final_linear

        # Rename for 2 reasons: keep the backup copy & also if symlink, don't overwrite the destination file
        fs_rename(layer_chkpt_path, layer_chkpt_path + ".orig")
        print("Saving {}".format(layer_chkpt_path))

        # Save the updated checkpoint
        torch.save(m, layer_chkpt_path)
        return dim

    def modify_optimizer_checkpoint(self, dim):
        name = "mp_rank_00_model_states.pt"
        layer_chkpt_path = os.path.join(self.checkpoint_path, name)
        checkpoint = torch.load(layer_chkpt_path)
        checkpoint["optimizer"]["fp32_groups_flat"] = [torch.zeros(dim * dim, dtype=torch.float32)]
        if 'args' in checkpoint and 'num_layers' in checkpoint['args']:
            checkpoint['args']['num_layers'] = self.new_layers_num

        # Rename for 2 reasons: keep the backup copy & also if symlink, don't overwrite the destination file
        fs_rename(layer_chkpt_path, layer_chkpt_path + ".orig")

        # Save the updated checkpoint
        print("Saving {}".format(name))
        torch.save(checkpoint, layer_chkpt_path)

def main():
    orig_model_path = sys.argv[1]
    orig_checkpoint = sys.argv[2]
    new_model_path = sys.argv[3]
    model_id = 0

    transform = model_transform(orig_model_path, orig_checkpoint, new_model_path, model_id)
    mutable = transform.link_new_model(transform.orig_layers_num)
    dim = mutable.modify_layer_checkpoint()
    mutable.modify_optimizer_checkpoint(dim)

if __name__ == "__main__":
    main()
