import glob
import os
import random

import torch

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
        model = models.VGGUnetModel(n_channels=1,
                                    n_classes=config.n_classes,
                                    lr=config.lr,
                                    bilinear=not config.learnable_upsamples,
                                    eval_batchsize=config.batch_size)
    elif config.model_name == 'UNet3D':
        model = models.UNet3DModel(n_channels=1,
                                   n_classes=config.n_classes,
                                   slice_size=config.slice_size,
                                   lr=config.lr)
    else:
        raise Exception("No such train method")

    return model


def train(config):
    random.seed(1)
    torch.manual_seed(1)

    model = get_model(config)
    model_dir = f"automated_train_dir/{os.path.basename(config.data_path)}/{config}"

    dataloaders = get_dataloaders(config)

    train_model(model, dataloaders, model_dir, config)


def test(config):
    model = get_model(config)
    model.to(config.device)
    model_dir = f"automated_train_dir/{os.path.basename(config.data_path)}/{config}"

    latest_ckpt = max(glob.glob(f'{model_dir}/*.pth'), key=os.path.getctime)
    model.load_state_dict(torch.load(latest_ckpt))

    config.batch_size = 1
    train_loader, val_loader = get_dataloaders(config)
    train_loader.dataset.transforms = val_loader.dataset.transforms # avoid slicing and use full volumes

    # print("Testing model...")
    train_report = evaluate(model, train_loader, config.device)
    validation_report = evaluate(model, val_loader, config.device)

    f = open(os.path.join(model_dir, "test_report.txt"), 'w')
    f.write(' , '.join(train_report.keys()) + "\n")
    f.write(' , '.join([f"{t:.3f}/{v:.3f}" for t, v in zip(train_report.values(), validation_report.values())]))
    f.close()
    # print(f"Train: {', '.join([f'Avg {k}: {v:.3f}' for k, v in train_report.items()])}")
    # print(f"Valid: {', '.join([f'Avg {k}: {v:.3f}' for k, v in validation_report.items()])}")


if __name__ == '__main__':
    exp_config = ExperimentConfigs(model_name='UNet3D', lr=0.001, augment_data=True, val_set='A', ignore_background=True, slice_size=16, batch_size=4)
    # for valset in ['A', 'B', "C"]:
    #        
    #     for config, config, config in [
    #         (ModelConfigs(model_name='UNet3D', lr=0.001), DataConfigs(augment_data=True, val_set=valset, ignore_background=True, slice_size=8, batch_size=8), TrainConfigs())
    #     ]:


    train(exp_config)
    test(exp_config)
