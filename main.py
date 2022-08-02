from pickle import load
from config.config import load_opts, save_opts

import os
opts=load_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id

from torch.utils.data import DataLoader
from torch import nn

from utils import gpu_initializer
import os 

import torch

from utils import colorize
import time

from train import trainer_dict
from data import dataset_dict
from models import model_dict

import numpy as np

import random 

from pathlib import Path
import glob

from multiprocessing import Pool

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from misc.evaluate_sub_alignment import eval_subtitle_alignment

from misc.postprocessing_remove_intersections import postprocessing_remove_intersections

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pp(test_file, opts=opts):
    return postprocessing_remove_intersections(test_file, 
    path_subtitles=opts.pr_sub_path, 
    path_probabilities=opts.save_probs_folder, 
    path_postpro_subs=opts.save_postpro_subs_folder)   

def main(opts):

    print('Cuda current device ', torch.cuda.current_device())

    set_seed(42)

    if opts.test_only: 
        assert opts.centre_window, 'Window should be fixed at test time, use option centre_window'
        assert not opts.jitter_location, 'Do not jitter location of prior at test time'
        assert opts.jitter_width_secs==0, 'Do not jitter location of prior at test time'
        assert opts.drop_feats==0, 'Do not drop features at test time'
        assert opts.shuffle_feats==0, 'Do not shuffle features at test time'
        assert opts.shuffle_words_subs==0, 'Do not shuffle subtitle words at test time'
        assert opts.drop_words_subs==0, 'Do not drop subtitle words at test time'

    if not opts.test_only:

        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id+int(time.time()))

        dataset = dataset_dict[opts.dataset](mode='train', opts=opts)
        dataloader = DataLoader(
            dataset,
            batch_size=opts.batch_size,
            shuffle=False,
            num_workers=opts.n_workers,
            worker_init_fn=worker_init_fn,
        )

        dataset_val = dataset_dict[opts.dataset](mode='val', opts=opts)
        dataloader_val = DataLoader(dataset_val,
                                    batch_size=opts.batch_size,
                                    shuffle=False,
                                    num_workers=opts.n_workers)

    else:
        dataset = dataset_dict[opts.dataset](mode='test', opts=opts)
        dataloader = DataLoader(dataset,
                                    batch_size=opts.batch_size,
                                    shuffle=False,
                                    num_workers=opts.n_workers)

    if len(dataloader)>0: 
        model = model_dict[opts.model](opts=opts, dataloader=dataloader)
        print("Model's state_dict:")

        trainer = trainer_dict[opts.trainer](model, opts)

        if opts.resume:
            trainer.load_checkpoint(opts.resume)

        if not opts.test_only:
            save_opts(opts, opts.save_path + "/args.txt")
        
        if not opts.test_only:
            scorefile = open(opts.save_path + "/scores.txt", "a+")

            res_val, best_metric = trainer.train(
                dataloader_val, mode='val', epoch=-1)  # initialize metric

            for epoch in range(opts.n_epochs):
                print('Epoch {:d}/{:d}'.format(epoch, opts.n_epochs))
                
                res_tr, _ = trainer.train(dataloader,
                                        mode='train',
                                        epoch=epoch)

                # -- evaluate
                with torch.no_grad():  # to save memory
                    if round(epoch/opts.save_every_n) == epoch/opts.save_every_n:
                        res_val, val_metric = trainer.train(dataloader_val,
                                                            mode='val',
                                                            epoch=epoch)

                        scorefile.write("{} | {}\n".format(res_tr, res_val))
                        scorefile.flush()

                        print('saving model ', "model_{:010d}.pt".format(trainer.global_step))
                        model_ckpt = "model_{:010d}.pt".format(trainer.global_step)
                        trainer.save_checkpoint(model_ckpt)
                        

            scorefile.close()
        else:
            res_val, val_metric = trainer.train(dataloader,
                                mode='test',
                                epoch=0)

    else:
        print('Length of dataloader is 0') 

    if opts.test_only and opts.save_vtt: 

        test_files = open(opts.test_videos_txt, "r").read().split('\n')
        if opts.random_subset_data < len(test_files):
            random.seed(opts.random_subset_data_seed)
            test_files = random.sample(test_files, opts.random_subset_data)

        if os.path.exists(os.path.join(opts.gt_sub_path, test_files[0] + '/signhd.vtt')):
            sub_ext = '/signhd.vtt'
        elif os.path.exists(os.path.join(opts.gt_sub_path, test_files[0] + '.vtt')):
            sub_ext = '.vtt'
        else: 
            print('cannot find file')

        gt_anno_paths = [Path(os.path.join(opts.gt_sub_path, p+sub_ext)) for p in test_files]
        
        if opts.dtw_postpro:
            with Pool(opts.n_workers) as pool:
                pool.map(pp, test_files)
                
            print("after DTW output")
            eval_str = eval_subtitle_alignment(
                pred_path_root=Path(f'{opts.save_postpro_subs_folder}'),
                gt_anno_path_root=Path(f'{opts.gt_sub_path}'),
                list_videos=test_files,
                fps=25,
            )

if __name__ == '__main__':
    opts = load_opts()     
    main(opts)
