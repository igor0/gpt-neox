import math
import os
import re
import shutil
import torch
import yaml


def small_init_init_method(dim):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010), using a normal distribution."""
    std = math.sqrt(2 / (5 * dim))

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_

def fs_link(src, dst):
    print("Linking '{}' <- '{}'".format(src, dst))
    os.symlink(src, dst)

def fs_mkdir(path):
    print("Creating directory '{}'".format(path))
    os.mkdir(path)

def fs_copy(src, dst):
    print("Copying '{}' to '{}'".format(src, dst))
    shutil.copy(src, dst)

def fs_rename(src, dst):
    print("Renaming '{}' to '{}'".format(src, dst))
    os.rename(src, dst)

def get_layer_file_name_str(layer_id, model_id):
    return "layer_{}-model_{}-model_states.pt".format(layer_id, model_id)

def get_layer_file_name(layer_id, model_id):
    return get_layer_file_name_str("{:02d}".format(layer_id), "{:02d}".format(model_id))

def get_mstate_file_name_str(state_id):
    return "mp_rank_{}_model_states.pt".format(state_id)

def get_mstate_file_name(state_id):
    return get_mstate_file_name_str("{:02d}".format(state_id))

def enable_extra_linear(args):
    return args.mode == 'extra_linear' or args.mode == 'out_linear_all'

def enable_pre_extra_linear(args):
    return args.mode == 'pre_extra_linear'

class model:
    def __init__(self, model_path):
        self.checkpoint_path = self._get_checkpoint_path(model_path)
        self._inspect_checkpoint()

    def _get_checkpoint_path(self, model_path):
        checkpoint_path = None
        lastest_file_path = os.path.join(model_path, "latest")
        try:
            with open(lastest_file_path, "r") as f:
                checkpoint_name = f.readline().strip()
                checkpoint_path = os.path.join(model_path, checkpoint_name)
        except FileNotFoundError:
            raise FileNotFoundError("Could not find the latest checkpoint file in '{}'".format(model_path))

        return checkpoint_path

    def _inspect_checkpoint(self):
        layer_name_regex = re.compile(get_layer_file_name_str(r'(\d\d)', r'(\d\d)'))
        mstate_name_regex = re.compile(get_mstate_file_name_str(r'(\d\d)'))

        max_layer_id = -1
        max_model_id = -1
        max_mstate_id = -1

        for x in os.listdir(self.checkpoint_path):
            m = layer_name_regex.match(x)
            if m is not None:
                model_id = int(m[2])
                layer_id = int(m[1])

                max_layer_id = max(max_layer_id, layer_id)
                max_model_id = max(max_model_id, model_id)

            m = mstate_name_regex.match(x)
            if m is not None:
                mstate_id = int(m[1])
                max_mstate_id = max(max_mstate_id, mstate_id)

        self.max_layer_id = max_layer_id
        self.max_model_id = max_model_id
        self.max_mstate_id = max_mstate_id

        extra_layers = 5 # layers other than the regular transformer layers
        self.layers_num = (self.max_layer_id + 1) - extra_layers

class model_transform:
    BASELINE_CHECKPOINT_NAME = "baseline"

    def __init__(self, args):
        self.args = args
        self.orig = model(args.orig_model_path)

    def link_new_model(self, new_layers_num):
        # Create the directory structure for the transformed model
        orig_config_path, new_config_path, new_checkpoint_path = self._create_dirs()

        # Read, modify and write the config file
        self._modify_config(orig_config_path, new_config_path, new_layers_num)

        # Write the 'latest' file for the transformed model
        self._write_latest_file()

        # Create checkpoint links

        self._link_layer(self.orig, 0, 0, new_checkpoint_path)

        for i in range(new_layers_num):
            orig_i = i + 2
            if self.args.mode == "pre_logit_lens" or self.args.mode == "pre_extra_linear":
                orig_i = orig_i + self.orig.layers_num - new_layers_num

            self._link_layer(self.orig, orig_i, i + 2, new_checkpoint_path)

        # Link the normalization layer
        self._link_layer(self.orig, self.orig.layers_num + 3, new_layers_num + 3, new_checkpoint_path)

        # Link the final head layer
        model_head = self.orig if self.args.head is None else model(self.args.head)
        self._link_layer(model_head, model_head.layers_num + 4, new_layers_num + 4, new_checkpoint_path)

        # Link the model states
        for mstate_id in range(self.orig.max_mstate_id + 1):
            mstate_file = get_mstate_file_name(mstate_id)
            orig_mstate_path = os.path.join(self.orig.checkpoint_path, mstate_file)
            new_mstate_path = os.path.join(new_checkpoint_path, mstate_file)
            fs_link(orig_mstate_path, new_mstate_path)

        return mutable_model(self.args, new_checkpoint_path, new_layers_num, new_layers_num + 4, self.orig.max_model_id, self.orig.max_mstate_id)

    def copy_head(self):
        # Create the directory for the new head
        orig_config_path, new_config_path, new_checkpoint_path = self._create_dirs()

        # Copy the config file, so that we have it preserved with the head
        fs_copy(orig_config_path, new_config_path)

        # Write the 'latest' file for the transformed model
        self._write_latest_file()

        # Copy the head
        if self.args.extra_linear_only:
            self._copy_extra_linear(self.orig, self.orig.layers_num + 4, self.orig.layers_num + 4, new_checkpoint_path)
        else:
            self._link_layer(self.orig, self.orig.layers_num + 4, self.orig.layers_num + 4, new_checkpoint_path, copy=True)

    def _create_dirs(self):
        configs_dir = "configs"
        config_file = "config.yml"

        # Figure out the paths

        new_checkpoint_path = os.path.join(self.args.new_model_path, self.BASELINE_CHECKPOINT_NAME)
        new_configs_path = os.path.join(self.args.new_model_path, configs_dir)
        new_config_path = os.path.join(new_configs_path, config_file)
        orig_config_path = os.path.join(self.args.orig_model_path, configs_dir, config_file)

        # Create the directories

        fs_mkdir(self.args.new_model_path)
        fs_mkdir(new_checkpoint_path)
        fs_mkdir(new_configs_path)
        return orig_config_path, new_config_path, new_checkpoint_path

    def _modify_config(self, orig_config_path, new_config_path, new_layers_num):
        with open(orig_config_path) as conf_file:
            conf = yaml.load(conf_file, Loader=yaml.FullLoader)

        conf['load'] = self.args.new_model_path
        conf['save'] = self.args.new_model_path
        conf['finetune'] = True
        conf['num-layers'] = new_layers_num

        if self.args.masterport is not None:
            conf['master_port'] = self.args.masterport + 29500

        if self.args.seq_length is not None:
            conf['seq-length'] = self.args.seq_length
            conf['max-position-embeddings'] = 2 * self.args.seq_length

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
        elif self.args.mode == 'in_linear_all':
            conf['pythia_train_only'] = r'attention\.query_key_value|mlp\.dense_h_to_4h'
        elif self.args.mode == 'all' or self.args.mode == 'all_100k':
            if 'pythia_train_only' in conf:
                del conf['pythia_train_only']
        else:
            conf['pythia_train_only'] = self.args.mode

        if enable_extra_linear(self.args):
            conf['pythia_extra_linear'] =  True
        if enable_pre_extra_linear(self.args):
            conf['pythia_pre_extra_linear'] =  True

        if self.args.mode == 'all_100k':
            train_iters = 100_000
        else:
            train_iters = 10_000

        if 'gradient_accumulation_steps' in conf:
            train_iters = train_iters // conf['gradient_accumulation_steps']

        conf['train-iters'] = train_iters
        conf['lr-decay-iters'] = train_iters
        if 'tokenizer_type' in conf and conf['tokenizer_type'] == 'HFTokenizer':
            conf['data-path'] = '/mnt/ssd-1/data/pile_20B_tokenizer/pile_20B_tokenizer_text_document'
        else:
            conf['data-path'] = '/mnt/ssd-1/data/pile_00/pile_00_text_document'
        if self.args.seed is not None:
            conf['seed'] = self.args.seed
        if self.args.num_gpus is not None:
            conf['num_gpus'] = self.args.num_gpus

        if self.args.mode == 'final_linear':
            # Ultimately, different LRs seem to work for different depths here.
            # For dense_small, I believe I used 6e-4 for depths 0..7 and 6e-5
            # for depths 8..12.
            conf['optimizer']['params']['lr'] = 0.00006

        if self.args.lr is not None:
            conf['optimizer']['params']['lr'] = self.args.lr
            conf['min_lr'] = self.args.lr / 10.0

        if self.args.predict is not None:
            conf['pythia_predict_special'] = self.args.predict
            #if self.args.predict == 'sink':
                #conf['optimizer']['params']['lr'] = 0.00006

        print("Writing to", new_config_path)
        with open(new_config_path, 'w') as f:
            yaml.dump(conf, f)

    def _link_layer(self, orig, orig_layer_id, new_layer_id, new_checkpoint_path, copy=False):
        for model_id in range(self.orig.max_model_id + 1):
            orig_name = get_layer_file_name(orig_layer_id, model_id)
            new_name = get_layer_file_name(new_layer_id, model_id)

            orig_layer_path = os.path.join(orig.checkpoint_path, orig_name)
            new_layer_path = os.path.join(new_checkpoint_path, new_name)

            if copy:
                fs_copy(orig_layer_path, new_layer_path)
            else:
                fs_link(orig_layer_path, new_layer_path)

    def _copy_extra_linear(self, orig, orig_layer_id, new_layer_id, new_checkpoint_path):
        for model_id in range(self.orig.max_model_id + 1):
            orig_name = get_layer_file_name(orig_layer_id, model_id)
            new_name = get_layer_file_name(new_layer_id, model_id)

            orig_layer_path = os.path.join(orig.checkpoint_path, orig_name)
            new_layer_path = os.path.join(new_checkpoint_path, new_name)

            ch = torch.load(orig_layer_path)
            keys_to_remove = []
            for k in ch.keys():
                if not k.startswith("extra_linear"):
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del ch[k]
            torch.save(ch, new_layer_path)


    def _write_latest_file(self):
        new_latest_file_path = os.path.join(self.args.new_model_path, "latest")
        print("Writing to", new_latest_file_path)
        with open(new_latest_file_path, 'w') as fh:
            fh.write(self.BASELINE_CHECKPOINT_NAME)



class mutable_model:
    def __init__(self, args, checkpoint_path, new_layers_num, max_layer_id, max_model_id, max_mstate_id):
        self.args = args
        self.checkpoint_path = checkpoint_path
        self.new_layers_num = new_layers_num
        self.max_layer_id = max_layer_id
        self.max_model_id = max_model_id
        self.max_mstate_id = max_mstate_id

    def modify_layer_checkpoint(self):
        # If there is a newly added extra_linear head, initialize it to identity
        if enable_extra_linear(self.args) and self.args.head is None:
            extra_linear = None
            model_parallelism = self.max_model_id + 1
            for model_id in range(model_parallelism):
                name = get_layer_file_name(self.max_layer_id, model_id)
                layer_chkpt_path = os.path.join(self.checkpoint_path, name)
                m = torch.load(layer_chkpt_path, map_location=torch.device('cpu'))

                # Calculate the dimensions of the extra_linear slice for this model partition
                final_linear = m["final_linear.weight"]
                dim = final_linear.shape[1]
                assert dim % model_parallelism == 0
                dim_slice_width = dim // model_parallelism
                dim_slice1 = model_id * dim_slice_width
                dim_slice2 = dim_slice1 + dim_slice_width

                if extra_linear == None:
                    extra_linear = torch.eye(dim)
                    if self.args.predict is not None or self.args.random_init == True:
                        # Use the small_init initialization method for predictions other than
                        # next-token. This didn't seem to be necessary for smaller models, but was
                        # needed for the 20B model.
                        small_init_init_method(dim)(extra_linear)


                # Add extra_linaer into the checkpoint
                del m["final_linear.weight"]
                m["extra_linear.weight"] = extra_linear[:,dim_slice1:dim_slice2].float()
                m["final_linear.weight"] = final_linear

                # Rename for 2 reasons: keep the backup copy & also if symlink, don't overwrite the destination file
                fs_rename(layer_chkpt_path, layer_chkpt_path + ".orig")
                print("Saving {}".format(layer_chkpt_path))

                # Save the updated checkpoint
                torch.save(m, layer_chkpt_path)
                del m

        if enable_pre_extra_linear(self.args) and self.args.head is None:
            pre_extra_linear = None
            model_parallelism = self.max_model_id + 1
            for model_id in range(model_parallelism):
                name = get_layer_file_name(0, model_id)
                layer_chkpt_path = os.path.join(self.checkpoint_path, name)
                m = torch.load(layer_chkpt_path, map_location=torch.device('cpu'))

                # Calculate the dimensions of the extra_linear slice for this model partition
                word_embeddings = m["word_embeddings.weight"]
                dim = word_embeddings.shape[1]
                assert dim % model_parallelism == 0
                dim_slice_width = dim // model_parallelism
                dim_slice1 = model_id * dim_slice_width
                dim_slice2 = dim_slice1 + dim_slice_width

                if pre_extra_linear == None:
                    pre_extra_linear = torch.eye(dim)
                    small_init_init_method(dim)(pre_extra_linear)


                # Add pre_extra_linaer into the checkpoint
                m["pre_extra_linear.weight"] = pre_extra_linear[:,dim_slice1:dim_slice2].float()

                # Rename for 2 reasons: keep the backup copy & also if symlink, don't overwrite the destination file
                fs_rename(layer_chkpt_path, layer_chkpt_path + ".orig")
                print("Saving {}".format(layer_chkpt_path))

                # Save the updated checkpoint
                torch.save(m, layer_chkpt_path)
                del m

    def modify_optimizer_checkpoint(self):
        for mstate_id in range(self.max_mstate_id + 1):
            name = get_mstate_file_name(mstate_id)
            layer_chkpt_path = os.path.join(self.checkpoint_path, name)
            checkpoint = torch.load(layer_chkpt_path, map_location=torch.device('cpu'))
            #checkpoint["optimizer"]["fp32_groups_flat"] = []
            if 'args' in checkpoint and 'num_layers' in checkpoint['args']:
                checkpoint['args']['num_layers'] = self.new_layers_num
            else:
                # No change needed
                continue

            # Rename for 2 reasons: keep the backup copy & also if symlink, don't overwrite the destination file
            fs_rename(layer_chkpt_path, layer_chkpt_path + ".orig")

            # Save the updated checkpoint
            print("Saving {}".format(name))
            torch.save(checkpoint, layer_chkpt_path)
            del checkpoint
