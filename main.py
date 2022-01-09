import glob
import os
import random

import torch

from datasets.ct_dataset import get_dataloaders
from evaluate import evaluate
from models.VGan.model import VGanModel
from train import train_model

import models

if __name__ == '__main__':
    random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    train_method = 'Unet'
    data_path = 'datasets/Cropped_Tumoers_Dataset-(L-1_mm-2)'
    resize = 128
    train_dir = f"train_dir/Liver-Data_{train_method}-{resize}"

    mode = "test"

    if train_method == 'Unet':
        batch_size = 32
        model = models.UnetModel(n_channels=1, n_classes=3, bilinear=True, device=device, eval_batchsize=batch_size)
        slice_size = 1
        train_steps = 50000
    elif train_method == 'AdSeg':
        model = models.AdSegModel(n_channels=1, n_classes=3, device=device)
        batch_size = 1
        slice_size = 1
        train_steps = 100000
    elif train_method == 'adSeg-semi':
        model = models.AdverserialSegSemi(n_channels=1, n_classes=3, device=device)
        batch_size = 1
        slice_size = 1
        train_steps = 100000
    elif train_method == 'VGan':
        model = models.VGanModel(n_channels=1, n_classes=3, device=device)
        batch_size = 1
        slice_size = 1
        train_steps = 100000
    elif train_method == 'Vnet':
        batch_size = 1
        slice_size = 16
        train_steps = 10000
        model = models.VnetModel(n_channels=batch_size, n_classes=3, slice_size=slice_size, device=device)
    else:
        raise Exception("No such train method")

    params = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    if mode == 'train':
        dataloaders = get_dataloaders(data_path, val_perc=0.1, params=params, slice_size=slice_size, resize=resize)
        train_model(model, dataloaders, device, train_steps=train_steps, train_dir=train_dir)

    if mode == 'test':
        latest_ckpt = max(glob.glob(f'{train_dir}/*.pth'), key=os.path.getctime)
        model.load_state_dict(torch.load(latest_ckpt))

        params['batch_size'] = 1
        train_loader, test_loader = get_dataloaders(data_path, val_perc=0.1, params=params, slice_size=slice_size, resize=resize)
        train_loader.dataset.transforms = test_loader.dataset.transforms

        validation_report = evaluate(model, test_loader, device, f"{train_dir}/test_outputs", n_plotted_volumes=None)
        validation_report = [f'Avg {k}: {v:".3f}' for k, v in validation_report.items()]
        print(f"Train: {', '.join(validation_report)}")

        train_report = evaluate(model, train_loader, device, f"{train_dir}/train_outputs", n_plotted_volumes=None)
        train_report = [f'Avg {k}: {v:.3f}' for k, v in train_report.items()]
        print(f"Train: {', '.join(train_report)}")






