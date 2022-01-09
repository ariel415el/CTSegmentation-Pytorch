import glob
import os
import random

import torch

from datasets.ct_dataset import get_dataloaders
from evaluate import test
from train import train_model

from models.Unet.model import UnetModel
from models.Vnet.model import VnetModel
from models.Adversarial_Learning_For_Semi_Supervised_Semantic_Segmentation.model import AdverserialSegSemi
from models.Semantic_Segmentation_using_Adversarial_Networks.model import AdSegModel


if __name__ == '__main__':
    random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    train_method = 'Unet'
    data_path = 'datasets/Cropped_Tumoers_Dataset-(L-1_mm-2)'
    resize = 128
    train_dir = f"train_dir/Liver-Data_{train_method}-{resize}"

    mode = "train"

    if train_method == 'Unet':
        model = UnetModel(n_channels=1, n_classes=3, bilinear=True, device=device)
        batch_size = 32
        slice_size = 1
        train_steps = 50000
    elif train_method == 'AdSeg':
        model = AdSegModel(n_channels=1, n_classes=3, device=device)
        batch_size = 1
        slice_size = 1
        train_steps = 100000
    elif train_method == 'adSeg-semi':
        model = AdverserialSegSemi(n_channels=1, n_classes=3, device=device)
        batch_size = 1
        slice_size = 1
        train_steps = 100000
    elif train_method == 'Vnet':
        batch_size = 1
        slice_size = 16
        train_steps = 10000
        model = VnetModel(n_channels=batch_size, n_classes=3, slice_size=slice_size, device=device)
    else:
        raise Exception("No such train method")

    params = dict(batch_size=batch_size, num_workers=4, pin_memory=False)
    dataloaders = get_dataloaders(data_path, val_perc=0.1, params=params, slice_size=slice_size, resize=resize)

    if mode == 'train':
        train_model(model, dataloaders, device, train_steps=train_steps, train_dir=train_dir)

    if mode == 'test':
        latest_ckpt = max(glob.glob(f'{train_dir}/*.pth'), key=os.path.getctime)
        model.load_state_dict(torch.load(latest_ckpt))

        train_loader, test_loader = dataloaders
        train_loader.dataset.transforms = test_loader.dataset.transforms

        train_score, train_speed = test(model, train_loader, device, f"{train_dir}/train_outputs")
        print(f"Train: Avg score: {train_score}, Inference Slice/sec: {train_speed}")

        test_score, test_speed = test(model, test_loader, device, f"{train_dir}/test_outputs")
        print(f"Test: Avg score: {test_score}, Inference Slice/sec: {test_speed}")





