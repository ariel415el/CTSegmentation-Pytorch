import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets import augmentations
from datasets.data_utils import get_data_pathes, read_volume

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

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


def get_transforms(slice_size, resize, augment_data):
    val_transforms = [
        augmentations.random_clip(-100, 400),
        # augmentations.HistogramEqualization(256),
        augmentations.Znormalization(),
        ToTensor(),
        augmentations.Resize(resize)
    ]

    train_transforms = [SliceVolume(slice_size=slice_size)]
    if augment_data:
        train_transforms += [
            augmentations.ElasticDeformation3D(sigma=7, p=0.1),
            augmentations.RandomCrop(p=1),
            augmentations.random_clip((-200, -50), (256, 1024)),
            # augmentations.HistogramEqualization(256),
            augmentations.Znormalization(),
            ToTensor(),
            augmentations.random_flips(p=1),
            augmentations.RandomAffine(p=0.3, degrees=(-45, 45), translate=(0,0.15), scale=(0.75, 1)),
            augmentations.Resize(resize),
            augmentations.random_noise(p=0.5, std_factor=0.25),
        ]
    else:
        train_transforms += val_transforms


    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def get_datasets(data_root, split_mode, slice_size, resize, augment_data):
    """
    Gather and split data paths and create datasets with according transforms
    param: split_mode: float for random split by percentage if validation cases or list of case numbers for the validation set
    """
    data_paths = get_data_pathes(data_root)
    random.shuffle(data_paths)

    # Split to train val
    if type(split_mode) == float:
        n_val = int(len(data_paths) * split_mode)
        print([os.path.basename(x[0]) for x in data_paths[:n_val]])
        train_paths, val_paths = data_paths[n_val:], data_paths[n_val:]
    else: # list of cases
        train_paths = []
        val_paths = []
        for ct_path, gt_path in data_paths:
            is_train=True
            for case_num in split_mode:
                if f"volume-{case_num}-" in ct_path:
                    is_train = False
                    break
            if is_train:
                train_paths.append((ct_path, gt_path))
            else:
                val_paths.append((ct_path, gt_path))

    train_transforms, val_transforms = get_transforms(slice_size, resize, augment_data)
    tarin_set = CTDataset(train_paths, transforms=train_transforms)
    val_set = CTDataset(val_paths, transforms=val_transforms)

    return tarin_set, val_set


def get_dataloaders(data_root, split_mode, params, slice_size, resize, augment_data):
    """
    Get dataloaders for training and evaluation.
    train_by_volume: 3d/2d training returns full CT volumes (batch_size, slices, H, W) or (batch_size, H, W)
    """
    train_set, val_set = get_datasets(data_root, split_mode, slice_size, resize, augment_data)

    train_loader = DataLoader(train_set, shuffle=True, **params)

    params['batch_size'] = 1
    val_loader = DataLoader(val_set, shuffle=True, **params)

    return train_loader, val_loader


class CTDataset(Dataset):
    """
    Dataset of entire CT volumes.
    """
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
        self.n_slices = np.sum(n_slices)
        print(f"Done loading {self.n_slices} slices in {len(self.cts)} volumes, "
              f"avg axial size volumes {np.mean(n_slices):.2f}, {n_dropped_volumes} volumes dropped")

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, i):
        sample = self.cts[i],  self.segs[i]
        if self.transforms:
            sample = self.transforms(sample)

        # TODO: Note that this is only for 2 classes
        gt = (sample[1] == 2).long()
        mask = (sample[1] != 0)
        # sample[0][~mask] = 0

        return {'ct':  sample[0], "gt":  gt, 'mask': mask, 'case_name': self.case_names[i]}
