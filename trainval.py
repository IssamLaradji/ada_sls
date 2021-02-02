import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import math
import itertools
import os
import sys
import pylab as plt
import exp_configs
import time
import numpy as np
import torch.nn as nn
from src import models
from src import datasets
from src import optimizers
from src import utils as ut
from src import metrics

import argparse

from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import default_collate

# cudnn.benchmark = True

from haven import haven_utils as hu
from haven import haven_results as hr
# from haven import haven_dropbox as hd
from haven import haven_chk as hc
import shutil

import pprint


def trainval(exp_dict, savedir_base, reset, metrics_flag=True, datadir=None, use_cuda=False):
    # bookkeeping
    # ---------------

    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)

    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    print(pprint.pprint(exp_dict))
    print('Experiment saved in %s' % savedir)

    # set seed
    # ==================
    seed = 42 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
        assert torch.cuda.is_available(), 'cuda is not, available please run with "-c 0"'
    else:
        device = 'cpu'

    print('Running on device: %s' % device)

    # Dataset
    # ==================
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict)

    train_loader = DataLoader(train_set,
                              drop_last=True,
                              shuffle=True,
                              sampler=None,
                              batch_size=exp_dict["batch_size"])

    # Load Val Dataset
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   train_flag=False,
                                   datadir=datadir,
                                   exp_dict=exp_dict)

    # Model
    # ==================
    model = models.get_model(exp_dict["model"],
                             train_set=train_set).to(device=device)

    # Choose loss and metric function
    loss_function = metrics.get_metric_function(exp_dict["loss_func"])

    # Load Optimizer
    # ==============
    n_batches_per_epoch = len(train_set)/float(exp_dict["batch_size"])
    opt = optimizers.get_optimizer(opt=exp_dict["opt"],
                                   params=model.parameters(),
                                   n_batches_per_epoch=n_batches_per_epoch,
                                   n_train=len(train_set),
                                   train_loader=train_loader,
                                   model=model,
                                   loss_function=loss_function,
                                   exp_dict=exp_dict,
                                   batch_size=exp_dict["batch_size"])

    # Checkpointing
    # =============
    score_list_path = os.path.join(savedir, "score_list.pkl")
    model_path = os.path.join(savedir, "model_state_dict.pth")
    opt_path = os.path.join(savedir, "opt_state_dict.pth")

    if os.path.exists(score_list_path) and os.path.exists(model_path):
        # resume experiment
        score_list = ut.load_pkl(score_list_path)
        model.load_state_dict(torch.load(model_path))
        opt.load_state_dict(torch.load(opt_path))
        s_epoch = score_list[-1]["epoch"] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Start Training
    # ==============
    n_train = len(train_loader.dataset)
    n_batches = len(train_loader)
    batch_size = train_loader.batch_size

    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        # Set seed
        seed = epoch + exp_dict['runs']
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)

        score_dict = {"epoch": epoch}

        # Validate
        # --------
        if metrics_flag:
            # 1. Compute train loss over train set
            score_dict["train_loss"] = metrics.compute_metric_on_dataset(model,
                                                                         train_set,
                                                                         metric_name=exp_dict["loss_func"],
                                                                         batch_size=exp_dict['batch_size'])

            # 2. Compute val acc over val set
            score_dict["val_acc"] = metrics.compute_metric_on_dataset(model, val_set,
                                                                      metric_name=exp_dict["acc_func"],
                                                                      batch_size=exp_dict['batch_size'])

        # Train
        # -----
        model.train()
        print("%d - Training model with %s..." %
              (epoch, exp_dict["loss_func"]))

        s_time = time.time()
        n_train = len(train_set)
        for batch in tqdm.tqdm(train_loader):
            opt.zero_grad()
            opt_step(exp_dict['opt']['name'], opt, model, batch, loss_function, device=device)
        e_time = time.time()

        # Record step size and batch size
        score_dict["step"] = opt.state.get(
            "step", 0) / int(n_batches_per_epoch)
        score_dict["step_size"] = opt.state.get("step_size", {})
        score_dict["step_size_avg"] = opt.state.get("step_size_avg", {})
        score_dict["n_forwards"] = opt.state.get("n_forwards", {})
        score_dict["n_backwards"] = opt.state.get("n_backwards", {})
        score_dict["grad_norm"] = opt.state.get("grad_norm", {})
        score_dict["batch_size"] = batch_size
        score_dict["train_epoch_time"] = e_time - s_time
        score_dict.update(opt.state["gv_stats"])

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report and save
        print(pd.DataFrame(score_list).tail())
        ut.save_pkl(score_list_path, score_list)
        ut.torch_save(model_path, model.state_dict())
        ut.torch_save(opt_path, opt.state_dict())
        print("Saved: %s" % savedir)

    return score_list



def opt_step(name, opt, model, batch, loss_function, device):
    indices = batch['meta']['indices']
    images, labels = batch["images"].to(device), batch["labels"].to(device)

    if (name in ["sgd_armijo", 'adaptive_first', 'l4', 'ali_g']):
        closure = lambda : loss_function(model, images, labels, backwards=False)
        loss = opt.step(closure)
                
    elif (name in ['sps']):
        closure = lambda : loss_function(model, images, labels, backwards=False)
        loss = opt.step(closure, batch)

    elif (name in ["adam", "adagrad", 'radam', 'plain_radam', 'adabound']):
        loss = loss_function(model, images, labels)
        loss.backward()
        opt.step()

    else:
        raise ValueError('%s optimizer does not exist' % name)
    
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)
    parser.add_argument('-c', '--use_cuda', type=int, default=0)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # run experiments
    for exp_dict in exp_list:
        # do trainval
        trainval(exp_dict=exp_dict,
                 savedir_base=args.savedir_base,
                 reset=args.reset,
                 datadir=args.datadir,
                 use_cuda=args.use_cuda)
