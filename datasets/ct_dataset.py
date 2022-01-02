import numpy as np
import os
import random

import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import augmentations


def read_volume(path):
    volume = sitk.ReadImage(path, sitk.sitkInt16)
    volume = sitk.GetArrayFromImage(volume)

    return volume

def get_transforms(volume_training):
    train_transforms = []
    if volume_training:
        train_transforms.append(augmentations.SliceVolume(slice_size=48))
    train_transforms += [
        augmentations.RandomCrop(),
        augmentations.Resize(128),
        augmentations.ToTensor()
    ]

    val_transforms = [
        augmentations.Resize(128),
        augmentations.ToTensor()
    ]
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def get_datasets(data_root, val_perc, train_by_volume):
    ct_files = sorted(os.listdir(os.path.join(data_root, 'ct')))
    ct_files = [os.path.join(data_root, 'ct', x) for x in ct_files]
    segmentation_files = sorted(os.listdir(os.path.join(data_root, 'seg')))
    segmentation_files = [os.path.join(data_root, 'seg', x) for x in segmentation_files]

    data_paths = list(zip(ct_files, segmentation_files))
    random.shuffle(data_paths)
    n_val = int(len(data_paths) * val_perc)

    train_transforms, val_transforms = get_transforms(train_by_volume)
    tarin_set = CTDataset(data_paths[n_val:], by_volume=train_by_volume, transforms=train_transforms)
    val_set = CTDataset(data_paths[:n_val], by_volume=True, transforms=val_transforms)

    return tarin_set, val_set


def get_dataloaders(data_root, val_perc, batch_size, train_by_volume):
    """
    Get dataloaders for training and evaluation.
    train_by_volume: 3d/2d training returns full CT volumes (batch_size, slices, H, W) or (batch_size, H, W)
    """
    train_set, val_set = get_datasets(data_root, val_perc, train_by_volume)

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=False, **loader_args)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    return train_loader, val_loader


class CTDataset(Dataset):
    def __init__(self, data_paths, by_volume, transforms=None):
        self.transforms = transforms
        self.cts = []
        self.segs = []
        for ct_path, seg_path in data_paths:
            self.cts.append(read_volume(ct_path) / 255)
            self.segs.append(read_volume(seg_path))

        if by_volume:
            print(f"Done loading {len(self.cts)} volumes")
        else:
            self.cts = np.concatenate(self.cts)
            self.segs = np.concatenate(self.segs)
            print(f"Done loading {len(self.cts)} slices")

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, i):
        sample = self.cts[i], self.segs[i]
        if self.transforms:
            sample = self.transforms(sample)

        return sample


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from utils import overlay
    torch.manual_seed(1)
    # Test train data
    tsfrms = [
        transforms.Compose([]),
        transforms.Compose([augmentations.RandomCrop(p=1)]),
        transforms.Compose([augmentations.Rescale(p=1)]),
        transforms.Compose([augmentations.RandomCrop(p=1), augmentations.Rescale(p=1)])
    ]

    for tsfrm in tsfrms:
        tsfrm.transforms.append(augmentations.ToTensor())

    train_loader, val_loader = get_dataloaders('../datasets/processed-data-(0.5,2)', 0.1, 1)

    # Apply each of the above transforms on sample.
    train_loader.dataset.transforms = None
    fig = plt.figure()
    for k in range(200, len(train_loader.dataset), 50):
        for i, tsfrm in enumerate(tsfrms):
            sample = train_loader.dataset[k]
            sample = sample[0], sample[1]

            transformed_sample = tsfrm(sample)
            transformed_sample = transformed_sample[0].unsqueeze(0), transformed_sample[1].unsqueeze(0)
            visulized_sample = overlay(*transformed_sample)
            visulized_sample = visulized_sample[0].numpy().transpose(1,2,0).astype(np.uint8)
            ax = plt.subplot(2, 2, i + 1)
            plt.tight_layout()
            ax.set_title(type(tsfrm).__name__)

            ax.imshow(visulized_sample)

        plt.show()