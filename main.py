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
from cnn_trainer import CNNTrainer

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
    model = get_model(config)

    dataloaders = get_dataloaders(config)

    trainer = CNNTrainer(config, model_dir, smooth_score_size=10)
    # trainer.try_load(os.path.join(model_dir, "trainer.pt"))

    logging.info('Training..')

    start = time()
    trainer.train_model(model, dataloaders)
    train_time = time() - start

    train_report = trainer.get_best_smoothed()
    train_report['Train-time'] = train_time

    return train_report


def test(config, model_dir, n_last_ckpts=3, outputs_dir=None):
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
    for i, ckpt_path in enumerate(latest_ckpts):
        model.load_state_dict(torch.load(ckpt_path))

        ckpt_name = 'Best' if 'best' in ckpt_path else f'{i}-latest'
        logging.info(f'Evaluating checkpoint-{ckpt_name}')
        train_report = evaluate(model, train_loader, config.device, outputs_dir=outputs_dir)
        validation_report = evaluate(model, val_loader, config.device, outputs_dir=outputs_dir)

        results.update({f"{ckpt_name}-{k}": f"{train_report[k]:.3f} / {validation_report[k]:.3f}" for k in train_report})

    return results


def run_single_experiment():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    exp_config = ExperimentConfigs(model_name='UNet3D', lr=0.00001, slice_size=32, batch_size=2,
                                   augment_data=True, Z_normalization=True, force_non_empty=True, ignore_background=True,
                                   train_steps=200000, eval_freq=1000)
    model_dir = f"/mnt/storage_ssd/train_dir/{os.path.basename(exp_config.data_path)}/{exp_config}"
    train(exp_config, model_dir)
    test_report = test(exp_config, model_dir, n_last_ckpts=1)
    logging.info(test_report)
    print(train_report)


def run_multiple_experiments():
    outputs_dir = "cluster_training"
    os.makedirs(outputs_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(outputs_dir, 'log-file.log'), format='%(asctime)s:%(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')

    full_report = pd.DataFrame()
    common_kwargs = dict(lr=0.00001, augment_data=True, Z_normalization=True, force_non_empty=True, ignore_background=True, batch_size=32, eval_freq=1000, train_steps=15000)
    for exp_config in [
            ExperimentConfigs(model_name='UNet', slice_size=1, **common_kwargs),
            ExperimentConfigs(model_name='VGGUNet', slice_size=1, **common_kwargs),
            ExperimentConfigs(model_name='VGGUNet2_5D', slice_size=3, **common_kwargs)
    ]:
        logging.info(f'#### {exp_config} ####')
        experiment_report = dict(Model_name=str(exp_config), N_slices=exp_config.train_steps * exp_config.batch_size * exp_config.slice_size)
        for val_set in ['A', 'B', 'C']:
            logging.info(f'# Validation set {val_set}')

            exp_config.val_set = val_set
            model_dir = f"{outputs_dir}/{exp_config}"

            train_report = train(exp_config, model_dir)

            test_report = test(exp_config.copy(), model_dir, n_last_ckpts=2)

            experiment_report.update({f"{k}-{val_set}": v for k,v in train_report.items()})
            experiment_report.update({f"{k}-{val_set}": v for k,v in run_report.items()})

        full_report = full_report.append(experiment_report, ignore_index=True)
        full_report.to_csv(os.path.join(outputs_dir, f"tmp-report.csv"), sep=',')

    full_report = full_report.set_index('Model_name')
    full_report.to_csv(os.path.join(outputs_dir, f"Final-report.csv"), sep=',')

    # ExperimentConfigs(model_name='UNet3D', lr=0.0001, augment_data=True,
    #                   ignore_background=True, slice_size=16, batch_size=2, eval_freq=100,
    #                   learnable_upsamples=True)

if __name__ == '__main__':
    random.seed(1)
    torch.manual_seed(1)
    run_single_experiment()
    # run_multiple_experiments()
