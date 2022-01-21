import glob
import os
import random

import torch

from datasets.ct_dataset import get_dataloaders
from evaluate import evaluate
from train import train_model

import models


def get_model(model_name, n_classes):
    if model_name == 'Unet':
        batch_size = 32
        model = models.UnetModel(n_channels=1, n_classes=n_classes, bilinear=True, device=device, eval_batchsize=batch_size)
        slice_size = 1
    elif model_name == 'AdSeg':
        model = models.AdSegModel(n_channels=1, n_classes=n_classes, device=device)
        batch_size = 1
        slice_size = 1
    elif model_name == 'adSeg-semi':
        model = models.AdverserialSegSemi(n_channels=1, n_classes=n_classes, device=device)
        batch_size = 1
        slice_size = 1
    elif model_name == 'VGan':
        batch_size = 32
        model = models.VGanModel(n_channels=1, n_classes=n_classes, device=device, eval_batchsize=batch_size)
        slice_size = 1
    elif model_name == 'Vnet':
        batch_size = 4
        slice_size = 16
        model = models.VnetModel(n_channels=1, n_classes=n_classes, slice_size=slice_size, device=device)
    else:
        raise Exception("No such train method")

    return model, slice_size, batch_size


if __name__ == '__main__':
    random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    mode = "train"
    model_name = 'Unet'
    n_classes = 2
    resize = 128
    augment_data = False
    ignore_background = False
    data_path = 'datasets/LiverData_(S-1_MS-(3, 10, 10)_Crop-CL-1_margins-(1, 1, 1)_OB-0.5_MD-11)'
    val_cases = [19, 76, 50, 92, 88, 122, 100, 71, 23, 28, 9, 119, 39]

    model, slice_size, batch_size = get_model(model_name, n_classes)
    model_dir = f"train_dir/{os.path.basename(data_path)}/{model_name}-{resize}{'_augment' if augment_data else ''}{'_mask_bg' if ignore_background else ''}"

    params = dict(batch_size=batch_size, num_workers=1)  # dataset loaded into memory no need for workers
    if mode == 'train':
        dataloaders = get_dataloaders(data_path, split_mode=val_cases, params=params, slice_size=slice_size, resize=resize, augment_data=augment_data)
        train_model(model, dataloaders, device, train_steps=100000, train_dir=model_dir, ignore_background=ignore_background)

    if mode == 'test':
        latest_ckpt = max(glob.glob(f'{model_dir}/*.pth'), key=os.path.getctime)
        model.load_state_dict(torch.load(latest_ckpt))

        params['batch_size'] = 1
        train_loader, test_loader = get_dataloaders(data_path, split_mode=0.1, params=params, slice_size=slice_size, resize=resize)
        train_loader.dataset.transforms = test_loader.dataset.transforms

        validation_report = evaluate(model, test_loader, device, f"{model_dir}/test_outputs", n_plotted_volumes=None)
        validation_report = [f'Avg {k}: {v:.3f}' for k, v in validation_report.items()]
        print(f"Valid: {', '.join(validation_report)}")

        train_report = evaluate(model, train_loader, device, f"{model_dir}/train_outputs", n_plotted_volumes=None)
        train_report = [f'Avg {k}: {v:.3f}' for k, v in train_report.items()]
        print(f"Train: {', '.join(train_report)}")






