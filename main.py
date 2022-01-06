import glob
import os
import random

import torch

from datasets.ct_dataset import get_dataloaders
from evaluate import test
from models.Unet.model import UnetModel, SliceLoss
from models.Vnet.model import VnetModel, VolumeLoss
from models.Semantic_Segmentation_using_Adversarial_Networks.model import AdSegModel
from train import train_model



if __name__ == '__main__':
    random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    train_method = 'Unet'
    # data_path = 'datasets/processed-data-(0.5,2)'
    data_path = 'datasets/Cropped_Tumoers_Dataset-(L-1_mm-2)'
    resize = 128
    train_dir = f"train_dir/Liver-Data_{train_method}-{resize}"

    mode = "test"

    if train_method == 'Unet':
        model = UnetModel(n_channels=1, n_classes=3, bilinear=True, lr=0.000001, device=device)
        slice_size = 1
        train_steps = 50000
    elif train_method == 'AdSeg':
        model = AdSegModel(n_channels=1, n_classes=3, lr=0.00001, device=device)
        slice_size = 1
        train_steps = 100000
    elif train_method == 'Vnet':
        model = VnetModel(n_channels=1, n_classes=3, d=16, lr=0.0001, device=device)
        slice_size = 16
        train_steps = 10000
    else:
        raise Exception("No such train method")

    if mode == 'train':
        dataloaders = get_dataloaders(data_path, val_perc=0.1, batch_size=1, slice_size=slice_size, resize=resize)
        train_model(model, dataloaders, device, train_steps=train_steps, train_dir=train_dir)

    if mode == 'test':
        latest_ckpt = max(glob.glob(f'{train_dir}/*.pth'), key=os.path.getctime)
        model.load_state_dict(torch.load(latest_ckpt))

        train_loader, test_loader = get_dataloaders(data_path, val_perc=0.1, batch_size=1, slice_size=slice_size,resize=resize)
        train_loader.dataset.transforms = None

        test_score, test_speed = test(model, test_loader, device, f"{train_dir}/test_outputs")
        train_score, train_speed = test(model, train_loader, device, f"{train_dir}/train_outputs")
        print(f"Test: Avg score: {test_score}, Inference Slice/sec: {test_speed}")
        print(f"Train: Avg score: {train_score}, Inference Slice/sec: {train_speed}")





