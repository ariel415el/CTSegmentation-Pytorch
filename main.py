import random

import torch

from datasets.ct_dataset import get_dataloaders
from evaluate import test
from models.Unet.model import UnetModel, SliceLoss
from models.Vnet.model import VnetModel, VolumeLoss
from train import train_model

if __name__ == '__main__':
    random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    data_path = 'datasets/processed-data-(0.5,2)'

    # model = UnetModel(n_channels=1, n_classes=3, bilinear=True)
    # criterion = SliceLoss
    # slice_size = 1
    # lr=0.000001
    # resize = 128
    # train_dir = "train_dir/Unet-128"
    # train_steps = 100000

    model = VnetModel(n_channels=1, n_classes=3, d=16)
    criterion = VolumeLoss
    slice_size = 16
    lr = 0.001
    resize = 256
    train_steps = 10000
    train_dir = "train_dir/Vnet-256-tversky-loss-0.5"

    dataloaders = get_dataloaders(data_path, val_perc=0.1, batch_size=1, slice_size=slice_size, resize=resize)
    train_model(model, criterion, dataloaders, device, lr=lr, train_steps=train_steps, train_dir=train_dir)

    # model.net.load_state_dict(torch.load('train_dir/Vnet-128/checkpoint_epoch9990.pth')['net'])
    # dataloader, _ = get_dataloaders(data_path, val_perc=0.1, batch_size=1, slice_size=slice_size, resize=resize)
    # test(model, dataloader, device, "test")


