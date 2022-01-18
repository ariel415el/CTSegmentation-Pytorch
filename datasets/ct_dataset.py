import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets import augmentations
from datasets.data_utils import get_data_pathes, read_volume


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, segmap = sample
        return torch.from_numpy(image), torch.from_numpy(segmap)


class SliceVolume(object):
    """Slice a portion of fixed size from the volume
      return a slice of size "slice_size" in the -4 dimension of the sample.
    """
    def __init__(self, slice_size=48):
        self.slice_size = slice_size

    def __call__(self, sample):
        image, segmap = sample

        if image.shape[-3] > self.slice_size:
            start_slice = np.random.randint(0, image.shape[-3] - self.slice_size)
            end_slice = start_slice + self.slice_size - 1

            image = image[..., start_slice:end_slice + 1, :, :]
            segmap = segmap[..., start_slice:end_slice + 1, :, :]

        return image, segmap


def get_transforms(slice_size, resize):
    train_transforms = [
        SliceVolume(slice_size=slice_size),
        augmentations.ElasticDeformation3D(sigma=7, p=0.1),
        augmentations.RandomCrop(p=1),
        augmentations.random_clip((-200, -50), (256, 1024)),
        ToTensor(),
        augmentations.Resize(resize),
        augmentations.random_flips(p=1),
        augmentations.random_noise(p=0.5, std=0.25),
    ]

    val_transforms = [
        ToTensor(),
        augmentations.random_clip(-100, 400),
        augmentations.Resize(resize)
    ]
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def get_datasets(data_root, val_perc, slice_size, resize):
    data_paths = get_data_pathes(data_root)
    random.shuffle(data_paths)

    # Split to train val
    train_transforms, val_transforms = get_transforms(slice_size, resize)

    n_val = int(len(data_paths) * val_perc)
    tarin_set = CTDataset(data_paths[n_val:], transforms=train_transforms)
    val_set = CTDataset(data_paths[:n_val], transforms=val_transforms)

    return tarin_set, val_set


def get_dataloaders(data_root, val_perc, params, slice_size=1, resize=128):
    """
    Get dataloaders for training and evaluation.
    train_by_volume: 3d/2d training returns full CT volumes (batch_size, slices, H, W) or (batch_size, H, W)
    """
    train_set, val_set = get_datasets(data_root, val_perc, slice_size, resize)

    train_loader = DataLoader(train_set, shuffle=True, **params)
    params['batch_size'] = 1
    val_loader = DataLoader(val_set, shuffle=True, **params)

    return train_loader, val_loader


class CTDataset(Dataset):
    def __init__(self, data_paths, transforms=None, min_n_slices=None):
        self.transforms = transforms
        self.cts = []
        self.segs = []
        self.case_names = []
        n_slices = []
        n_dropped_volumes = 0
        print("Loading data into memory... ", end='')
        for ct_path, seg_path in data_paths:
            label_map = read_volume(seg_path)
            if min_n_slices is None or label_map.shape[0] >= min_n_slices:
                self.segs.append(label_map.astype(np.uint8))
                self.cts.append(read_volume(ct_path))
                self.case_names.append(os.path.splitext(os.path.basename(ct_path))[0])
                n_slices.append(self.cts[-1].shape[-3])
            else:
                n_dropped_volumes += 1

        print(f"Done loading {np.sum(n_slices)} slices in {len(self.cts)} volumes, "
              f"avg axial size volumes {np.mean(n_slices):.2f}, {n_dropped_volumes} volumes dropped")

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, i):
        sample = self.cts[i],  self.segs[i]
        if self.transforms:
            sample = self.transforms(sample)

        gt = (sample[1] == 2).long()
        mask = (sample[1] != 0)

        return {'ct':  sample[0], "gt":  gt, 'mask': mask, 'case_name': self.case_names[i]}
