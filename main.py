import glob
import logging
import os
import random
import sys

import pandas as pd
import torch
from time import time

from datetime import datetime
from datasets.ct_dataset import get_dataloaders
from evaluate import evaluate
from train import train_model

import models
from config import *


def get_model(config):
    if config.model_name == 'UNet':
        model = models.UnetModel(n_channels=1,
                                 n_classes=config.n_classes,
                                 lr=config.lr,
                                 bilinear=not config.learnable_upsamples,
                                 eval_batchsize=config.batch_size)
    elif config.model_name == 'VGGUNet':
        model = models.VGGUnetModel(n_classes=config.n_classes,
                                    lr=config.lr,
                                    bilinear=not config.learnable_upsamples,
                                    eval_batchsize=config.batch_size)
    elif config.model_name == 'VGGUNet2_5D':
        model = models.VGGUnet2_5DModel(n_classes=config.n_classes,
                                        lr=config.lr,
                                        bilinear=not config.learnable_upsamples,
                                        eval_batchsize=config.batch_size)
    elif config.model_name == 'UNet3D':
        model = models.UNet3DModel(n_classes=config.n_classes,
                                   trilinear=not config.learnable_upsamples,
                                   slice_size=config.slice_size,
                                   lr=config.lr)
    else:
        raise Exception("No such train method")

    return model


def train(config, model_dir):
    random.seed(1)
    torch.manual_seed(1)

    model = get_model(config)

    dataloaders = get_dataloaders(config)

    logging.info('Training..')
    train_model(model, dataloaders, model_dir, config)


def test(config, model_dir, n_last_ckpts=3):

    # get dataloaders
    config.batch_size = 1
    train_loader, val_loader = get_dataloaders(config)
    train_loader.dataset.transforms = val_loader.dataset.transforms  # avoid slicing and use full volumes

    # get model
    model = get_model(config)
    model.to(config.device)

    # get checkpint paths
    latest_ckpts = sorted(glob.glob(f'{model_dir}/*.pth'), key=os.path.getctime)[-n_last_ckpts:]
    if f'{model_dir}/best.pth' not in latest_ckpts:
        latest_ckpts.append(f'{model_dir}/best.pth')
    results = dict()
    for ckpt_path in latest_ckpts:
        model.load_state_dict(torch.load(ckpt_path))

        ckpt_name = os.path.basename(os.path.splitext(ckpt_path)[0])
        logging.info(f'Evaluating checkpoint-{ckpt_name}')
        train_report = evaluate(model, train_loader, config.device)
        validation_report = evaluate(model, val_loader, config.device)

        results.update({f"{ckpt_name}-{k}": f"{train_report[k]:.3f} / {validation_report[k]:.3f}" for k in train_report})

    return results


def run_single_experiment():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    exp_config = ExperimentConfigs(model_name='UNet', slice_size=1, augment_data=True, Z_normalization=True, train_steps=200, eval_freq=100)
    model_dir = f"train_dir/{os.path.basename(exp_config.data_path)}/{exp_config}"
    train(exp_config, model_dir)
    train_report, validation_report = test(exp_config, model_dir, n_last_ckpts=1)
    logging.info({f"{k}": f"{train_report[k]:.3f} / {validation_report[k]:.3f}" for k in train_report})
    print(train_report)


def run_multiple_experiments():
    outputs_dir = "cluster_training"
    os.makedirs(outputs_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(outputs_dir, 'log.log'), format='%(asctime)s:%(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')

    full_report = pd.DataFrame()
    common_kwargs = dict(lr=0.00001, augment_data=True, Z_normalization=True, force_non_empty=True, ignore_background=True, batch_size=32, eval_freq=5, train_steps=10)
    for exp_config in [
            ExperimentConfigs(model_name='UNet', slice_size=1, **common_kwargs),
            ExperimentConfigs(model_name='VGGUNet', slice_size=1, **common_kwargs),
            # ExperimentConfigs(model_name='VGGUNet2_5D', slice_size=3, **common_kwargs)
    ]:
        experiment_report = dict(Model_name=str(exp_config), N_slices=exp_config.train_steps * exp_config.batch_size * exp_config.slice_size)
        for val_set in ['A', 'B']:#, 'C']:
            exp_config.val_set = val_set
            model_dir = f"{outputs_dir}/{exp_config}"

            start = time()
            train(exp_config, model_dir)
            train_time = time() - start

            run_report = test(exp_config, model_dir, n_last_ckpts=2)

            experiment_report.update({f"{k}-{val_set}": v for k,v in run_report.items()})
            experiment_report[f"Train-Time-{val_set}"] = train_time

        full_report = full_report.append(experiment_report, ignore_index=True)
        full_report.to_csv(os.path.join(outputs_dir, f"tmp-report.csv"), sep=',')

    full_report = full_report.set_index('Model_name')
    full_report.to_csv(os.path.join(outputs_dir, f"Final-report.csv"), sep=',')

    # ExperimentConfigs(model_name='UNet3D', lr=0.0001, augment_data=True,
    #                   ignore_background=True, slice_size=16, batch_size=2, eval_freq=100,
    #                   learnable_upsamples=True)

if __name__ == '__main__':
    # run_single_experiment()
    run_multiple_experiments()
