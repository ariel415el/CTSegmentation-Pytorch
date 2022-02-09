import logging
import os
import random
import sys
import pandas as pd
import json

from datasets.ct_dataset import get_dataloaders
from evaluate import evaluate
from cnn_trainer import CNNTrainer

import models
from config import *
from metrics import VolumeLoss


def get_model(config):
    if config.model_name == 'UNet':
        model = models.UnetModel(n_channels=1,
                                 n_classes=config.n_classes,
                                 p=64,
                                 lr=config.starting_lr,
                                 bilinear_upsample=not config.learnable_upsamples,
                                 eval_batchsize=32)
    elif config.model_name == 'VGGUNet':
        model = models.VGGUnetModel(n_classes=config.n_classes,
                                    lr=config.starting_lr,
                                    bilinear_upsample=not config.learnable_upsamples,
                                    eval_batchsize=32)
    elif config.model_name == 'VGGUNet2_5D':
        assert config.slice_size ==3
        model = models.VGGUnet2_5DModel(n_classes=config.n_classes,
                                        lr=config.starting_lr,
                                        bilinear_upsample=not config.learnable_upsamples,
                                        eval_batchsize=32)
    elif config.model_name == 'UNet3D':
        model = models.UNet3DModel(n_classes=config.n_classes,
                                   trilinear_upsample=not config.learnable_upsamples,
                                   slice_size=config.slice_size,
                                   p=32,
                                   lr=config.starting_lr)

    elif config.model_name == 'DARN':
        model = models.DARNModel(n_classes=config.n_classes,
                                   trilinear_upsample=not config.learnable_upsamples,
                                   slice_size=config.slice_size,
                                   p=8,
                                   lr=config.starting_lr)

    elif config.model_name == 'HeavyUNet':
        model = models.HeavyUnetModel(n_channels=1,
                                 n_classes=config.n_classes,
                                 lr=config.starting_lr,
                                 eval_batchsize=32)
    elif config.model_name == 'ResUNet':
        model = models.ResUnetModel(n_channels=1,
                                      n_classes=config.n_classes,
                                      lr=config.starting_lr,
                                      eval_batchsize=32)
    elif config.model_name == 'RecurrentUNet':
        model = models.RecurrentUnetModel(n_channels=1,
                                      n_classes=config.n_classes,
                                      lr=config.starting_lr,
                                      eval_batchsize=32)
    elif config.model_name == 'Res2Unet':
        model = models.Res2UnetModel(n_channels=1,
                                      n_classes=config.n_classes,
                                      lr=config.starting_lr,
                                      eval_batchsize=32)


    else:
        raise Exception("No such train method")

    return model


def train(exp_config, outputs_dir):
    model = get_model(exp_config.get_model_config())
    dataloaders = get_dataloaders(exp_config.get_data_config())
    trainer = CNNTrainer(exp_config.get_train_configs())

    model_dir = f"{outputs_dir}/{exp_config}"

    # copy config file
    exp_config.write_to_file(model_dir)

    # Try to load checkpoint
    if os.path.exists(os.path.join(model_dir, "latest.pth")):
        logging.info("Starting from latest checkpoint")
        ckpt = torch.load(os.path.join(model_dir, "latest.pth"), map_location=torch.device("cpu"))
        trainer.load_state(ckpt['trainer'])
        model.load_state_dict(ckpt['model'])

    trainer.train_model(model, dataloaders, model_dir)
    train_report = trainer.get_report()

    return train_report


def test(model_dir, ckpt_name='best'):
    """
    Test the model in a checkpoint on the entire dataset.
    """
    ckpt = torch.load(f'{model_dir}/{ckpt_name}.pth')

    config = json.load(open(f"{model_dir}.exp_configs.json"))

    # get dataloaders
    config.batch_size = 1
    train_loader, val_loader, _ = get_dataloaders(config.get_data_config())
    train_loader.dataset.transforms = val_loader.dataset.transforms  # avoid slicing and use full volumes

    # get model
    model = get_model(config.get_model_config())
    model.load_state_dict(ckpt['model'])
    model.to(config.device)

    volume_crieteria = VolumeLoss(config.dice_loss_weight, config.wce_loss_weight, config.ce_loss_weight)
    train_report = evaluate(model, train_loader, config.device, volume_crieteria, outputs_dir=os.path.join(model_dir, f'ckpt-{ckpt_name}', "train_debug"))
    validation_report = evaluate(model, val_loader, config.device, volume_crieteria, outputs_dir=os.path.join(model_dir, f'ckpt-{ckpt_name}', "val_debug"))

    report = pd.DataFrame()
    report = report.append({f"{ckpt_name}-{k}": f"{train_report[k]:.3f} / {validation_report[k]:.3f}" for k in train_report}, ignore_index=True)
    report.to_csv(os.path.join(model_dir, f'ckpt-{ckpt_name}', f"Test-Report.csv"), sep=',')


def run_single_experiment(outputs_dir):
    """
    Run a single experiment
    """
    os.makedirs(outputs_dir, exist_ok=True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    exp_config = ExperimentConfigs(model_name='UNet3D', starting_lr=0.00001, batch_size=4, slice_size=32,
                                   train_steps=100000, decay_steps=25000, decay_factor=0.1,
                                   wce_loss_weight=0, dice_loss_weight=0, ce_loss_weight=1,
                                   augment_data=True, ignore_background=True, force_non_empty=True,
                                   num_workers=2, eval_freq=1000)

    train_report = train(exp_config, outputs_dir)

    logging.info(train_report)


def run_multiple_experiments(outputs_dir):
    """
    Run multiple models with common configs on multple test sets and produce a report.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(outputs_dir, 'log-file.log'), format='%(asctime)s:%(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')

    full_report = pd.DataFrame()
    common_kwargs = dict(starting_lr=0.00001, batch_size=32, num_workers=2, train_steps=20000,
                         augment_data=True, ignore_background=True, force_non_empty=True,
                         dice_loss_weight=0, wce_loss_weight=0, ce_loss_weight=1)
    for exp_config in [
        ExperimentConfigs(model_name='HeavyUNet', elastic_deformations=False, **common_kwargs),
        ExperimentConfigs(model_name='ResUNet', elastic_deformations=False, **common_kwargs),
        # ExperimentConfigs(model_name='RecurrentUNet', elastic_deformations=False, **common_kwargs),
        # ExperimentConfigs(model_name='Res2Unet', elastic_deformations=False, **common_kwargs),
    ]:
        logging.info(f'#### {exp_config} ####')
        experiment_report = dict(Model_name=str(exp_config), N_slices=exp_config.train_steps * exp_config.batch_size * exp_config.slice_size)
        for val_set in ['A']:
            logging.info(f'# Validation set {val_set}')
            exp_config.val_set = val_set

            train_report = train(exp_config, outputs_dir)

            experiment_report.update({f"{k}-{val_set}": v for k,v in train_report.items()})

        full_report = full_report.append(experiment_report, ignore_index=True)
        full_report.to_csv(os.path.join(outputs_dir, f"tmp-report.csv"), sep=',')

    full_report = full_report.set_index('Model_name')
    full_report.to_csv(os.path.join(outputs_dir, f"Final-report.csv"), sep=',')


if __name__ == '__main__':
    outputs_root = '/mnt/storage_ssd/train_outputs'
    random.seed(1)
    torch.manual_seed(1)
    # run_single_experiment(f"{outputs_root}/train_dir")
    run_multiple_experiments(f"{outputs_root}/cluster_training_batch-4-heavyNets")
    # test('/mnt/storage_ssd/train_outputs/cluster_training_compare_losses_with_aug/UNet(p=64,BUS)__Aug_MaskBg_ZNorm_FNE_Loss(0.0Dice+1.0WCE+0.0CE)_V-A', ckpt_name='step-30000')
