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
    if path.endswith("nii"):
        volume = sitk.ReadImage(path, sitk.sitkInt16)
        volume = sitk.GetArrayFromImage(volume)
    else:
        volume = np.load(path)
    return volume


def get_transforms(slice_size, resize=128):
    train_transforms = [
        augmentations.SliceVolume(slice_size=slice_size),
        augmentations.RandomCrop(),
        augmentations.Resize(resize),
        augmentations.ToTensor()
    ]

    val_transforms = [
        augmentations.Resize(resize),
        augmentations.ToTensor()
    ]
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def get_datasets(data_root, val_perc, slice_size, resize):
    ct_files = sorted(os.listdir(os.path.join(data_root, 'ct')))
    ct_files = [os.path.join(data_root, 'ct', x) for x in ct_files]
    segmentation_files = sorted(os.listdir(os.path.join(data_root, 'seg')))
    segmentation_files = [os.path.join(data_root, 'seg', x) for x in segmentation_files]

    data_paths = list(zip(ct_files, segmentation_files))
    random.shuffle(data_paths)
    n_val = int(len(data_paths) * val_perc)

    train_transforms, val_transforms = get_transforms(slice_size, resize)
    tarin_set = CTDataset(data_paths[n_val + 25:], transforms=train_transforms)
    val_set = CTDataset(data_paths[:n_val], transforms=val_transforms)

    return tarin_set, val_set


def get_dataloaders(data_root, val_perc, params, slice_size=1, resize=128):
    """
    Get dataloaders for training and evaluation.
    train_by_volume: 3d/2d training returns full CT volumes (batch_size, slices, H, W) or (batch_size, H, W)
    """
    train_set, val_set = get_datasets(data_root, val_perc, slice_size, resize)

    val_loader = DataLoader(val_set, shuffle=True, **params)
    train_loader = DataLoader(train_set, shuffle=True, **params)

    return train_loader, val_loader


class CTDataset(Dataset):
    def __init__(self, data_paths, transforms=None):
        self.transforms = transforms
        self.cts = []
        self.segs = []
        self.case_names = []
        n_slices = 0
        n_dropped_volumes = 0
        for ct_path, seg_path in data_paths:
            label_map = read_volume(seg_path)
            if label_map.shape[0] >= 16:
                self.segs.append(label_map)
                self.cts.append(read_volume(ct_path) / 255)
                self.case_names.append(os.path.splitext(os.path.basename(ct_path)))
                n_slices += self.cts[-1].shape[-3]
            else:
                n_dropped_volumes += 1

        print(f"Done loading {n_slices} slices in {len(self.cts)} volumes, {n_dropped_volumes} volumes dropped")

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, i):
        sample = self.cts[i], self.segs[i]
        if self.transforms:
            sample = self.transforms(sample)

        return sample