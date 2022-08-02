import os
import subprocess

import torch
from tensorboardX import SummaryWriter
from transformers import Adafactor


from utils import colorize
import numpy as np


class BaseTrainer:
    """
    Base trainer class. Includes boilerplate code for:
      - Creating optimizer
      - Creating checkpoint and log directories
      - Saving and loading model checkpoints
    """

    def __init__(self, model, opts):
        # super(SlidingTrain, self).__init__()

        self.model = model
        self.opts = opts

        if opts.optimizer == 'adam':
            self.optimizer = torch.optim.Adam([
                {
                    'params': self.model.parameters(),
                    'lr': opts.lr, 
                },
            ])
        elif opts.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW([
                {
                    'params': self.model.parameters(),
                    'lr': opts.lr, 
                },
            ])
        elif opts.optimizer == 'adafactor':
            self.optimizer = Adafactor(self.model.parameters(),lr=1e-3,
                      eps=(1e-30, 1e-3),
                      clip_threshold=1.0,
                      decay_rate=-0.8,
                      beta1=None,
                      weight_decay=0.0,
                      relative_step=False,
                      scale_parameter=False,
                      warmup_init=False)
        else: 
            print('choose optimizer adam or adamw or adafactor')
        print(colorize('%s' % self.optimizer, 'green'))

        # TODO: Change this for multi-gpu training
        # self.device = torch.device('cuda:{}'.format(opts.gpu_id))
        self.device = torch.device('cuda:0')
        self.model.to(self.device)

        # --- make directories
        self.global_step = 0
        tb_path_train = os.path.join(opts.save_path, 'tb_logs', 'train')
        tb_path_val = os.path.join(opts.save_path, 'tb_logs', 'val')
        saved_vids_path = os.path.join(opts.save_path, 'saved_vids')

        self.tb_fps = 25

        self.checkpoints_path = opts.save_path + "/checkpoints"
        if not (os.path.exists(self.checkpoints_path)):
            os.makedirs(self.checkpoints_path)

        # if os.path.exists(opts.save_path):
        #     #   Clean up old tb logs
        #     command = 'rm %s/* -rf' % (tb_path_train)
        #     print(colorize(command, 'gray'))
        #     subprocess.call(command, shell=True, stdout=None)

        #     command = 'rm %s/* -rf' % (tb_path_val)
        #     print(colorize(command, 'gray'))
        #     subprocess.call(command, shell=True, stdout=None)

        #     command = 'rm %s/* -rf' % (saved_vids_path)
        #     print(colorize(command, 'gray'))
        #     subprocess.call(command, shell=True, stdout=None)

        # ------------- set up tb saver ----------------
        try:
            os.makedirs(opts.save_path)
        except:
            pass
        self.tb_writer = SummaryWriter(tb_path_train)

    def save_checkpoint(self, ckpt_name):
        save_dict = {
            'state_dict': self.model.state_dict(),
            'global_step': self.global_step,
        }
        torch.save(save_dict, os.path.join(self.checkpoints_path, ckpt_name))

    def load_checkpoint(self, ckpt_paths):

        for chkpt in ckpt_paths:

            import glob
            if chkpt.endswith('.pt'):
                ckpt = chkpt
            else:
                checkpoints = glob.glob('{}/checkpoints/*'.format(chkpt))
                checkpoints.sort()
                if len(checkpoints) > 0:
                    ckpt = checkpoints[-1]  # load the last one
                else:
                    assert 0, "No models found in {}!".format(ckpt_paths)

            loaded_state = torch.load(
                ckpt, map_location=lambda storage, loc: storage)
            if 'state_dict' in loaded_state:  # means we have dict in dict
                if 'global_step' in loaded_state:
                    self.global_step = loaded_state['global_step']
                loaded_state = loaded_state['state_dict']

            self.load_model_params(self.model, loaded_state)

            self.ckpt = ckpt
            print(colorize("Model {} loaded!".format(ckpt), 'green'))

    def load_model_params(self, model, loaded_state):

        self_state = model.state_dict()

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print(colorize("%s is not in the model." % origname, 'red'))
                    continue

            if self_state[name].size() != param.size():
                if np.prod(param.shape) == np.prod(self_state[name].shape):
                    print(
                        colorize(
                            "Caution! Parameter length: {}, model: {}, loaded: {}, Reshaping"
                            .format(origname, self_state[name].shape,
                                    loaded_state[origname].shape), 'red'))
                    param = param.reshape(self_state[name].shape)
                else:
                    print(
                        colorize(
                            "Wrong parameter length: {}, model: {}, loaded: {}".
                            format(origname, self_state[name].shape,
                                   loaded_state[origname].shape), 'red'))
                    continue

            self_state[name].copy_(param)


def train(self):
    raise NotImplementedError
