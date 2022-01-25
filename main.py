import glob
import os
import random

import torch

from datasets.ct_dataset import get_dataloaders
from evaluate import evaluate
from train import train_model

import models
import config


def get_model(model_config):
    if model_config.model_name == 'UNet':
        model = models.UnetModel(n_channels=1,
                                 n_classes=model_config.n_classes,
                                 lr=model_config.lr,
                                 bilinear=not model_config.learnable_upsamples,
                                 eval_batchsize=model_config.batch_size)
    elif model_config.model_name == 'VGGUNet':
        model = models.VGGUnetModel(n_channels=1,
                                    n_classes=model_config.n_classes,
                                    lr=model_config.lr,
                                    bilinear=not model_config.learnable_upsamples,
                                    eval_batchsize=model_config.batch_size)
    # elif config.model_name == 'UNet3D':
    #     model = models.UNet3DModel(n_channels=1,
    #                                n_classes=config.n_classes,
    #                                slice_size=config.slice_size,
    #                                lr=config.lr)
    else:
        raise Exception("No such train method")

    return model


def train(model_config, data_config, train_config):
    random.seed(1)
    torch.manual_seed(1)

    model = get_model(model_config)
    model_dir = config.compose_experiment_name(model_config, data_config, train_config)

    dataloaders = get_dataloaders(data_config)
    train_model(model, dataloaders, model_dir, train_config)


def test(model_config, data_config, train_config):
    model = get_model(model_config)
    model.to(train_config.device)
    model_dir = config.compose_experiment_name(model_config, data_config, train_config)

    latest_ckpt = max(glob.glob(f'{model_dir}/*.pth'), key=os.path.getctime)
    model.load_state_dict(torch.load(latest_ckpt))

    data_config.batch_size = 1
    train_loader, val_loader = get_dataloaders(data_config)
    train_loader.dataset.transforms = val_loader.dataset.transforms # avoid slicing and use full volumes

    print("Testing model...")
    train_report = evaluate(model, train_loader, train_config.device)
    validation_report = evaluate(model, val_loader, train_config.device)

    f = open(os.path.join(model_dir, "test_report.txt"), 'w')
    f.write(' , '.join(train_report.keys()) + "\n")
    f.write(' , '.join([f"{t:.3f}/{v:.3f}" for t, v in zip(train_report.values(), validation_report.values())]))
    f.close()
    print(f"Train: {', '.join([f'Avg {k}: {v:.3f}' for k, v in train_report.items()])}")
    print(f"Valid: {', '.join([f'Avg {k}: {v:.3f}' for k, v in validation_report.items()])}")


if __name__ == '__main__':
    model_config = config.ModelConfigs()
    train_config = config.TrainConfigs()
    for valset in ['A', 'B', "C"]:
        for data_config in [
            config.DataConfigs(val_set=valset),
            config.DataConfigs(val_set=valset, augment_data=True),
            config.DataConfigs(val_set=valset, augment_data=True, learnable_upsamples=True),
            config.DataConfigs(val_set=valset, augment_data=True, delete_background=True),
            config.DataConfigs(val_set=valset, augment_data=True, Z_normalization=True),
            config.DataConfigs(val_set=valset, augment_data=True, hist_equalization=True),
        ]:
            model_config.batch_size = data_config.batch_size
            train(model_config, data_config, train_config)
            test(model_config, data_config, train_config)