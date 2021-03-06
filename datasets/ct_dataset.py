import logging
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


# NO_TUMOR_CTS = [87, 47, 105, 106, 119, 114, 41, 38, 34, 115, 91, 32, 89]

LITS2017_VALSETS = {'A': [19, 76, 50, 92, 88, 122, 100, 71, 23, 28, 9, 119, 39],  # n_empty: 1
                    'B': [101, 99, 112, 107, 24, 34, 30, 120, 90, 98, 118, 83, 0],  # n_empty: 1
                    'C': [97, 5, 17, 41, 105, 57, 15, 110, 93, 106, 32, 124, 68]}  # n_empty: 4


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, segmap = sample
        image, segmap = torch.from_numpy(image), torch.from_numpy(segmap.astype(np.uint8))
        return image, segmap


class SliceVolume(object):
    """Slice a portion of fixed size from the volume
      return a slice of size "slice_size" in the -4 dimension of the sample.
    """

    def __init__(self, slice_size=48, force_non_empty=0):
        self.slice_size = slice_size
        self.force_non_empty = force_non_empty

    def __call__(self, sample):
        image, segmap = sample

        if image.shape[-3] > self.slice_size:
            is_empty = np.all(segmap == 0)
            start_slice = np.random.randint(0, image.shape[-3] - self.slice_size)
            end_slice = start_slice + self.slice_size - 1

            if self.force_non_empty > 0 and not is_empty and torch.rand(1) < self.force_non_empty:
                # sample only non-empty slices
                while np.all(segmap[..., start_slice:end_slice + 1, :, :] == 0):
                    start_slice = np.random.randint(0, image.shape[-3] - self.slice_size)
                    end_slice = start_slice + self.slice_size - 1

            image = image[..., start_slice:end_slice + 1, :, :]
            segmap = segmap[..., start_slice:end_slice + 1, :, :]

        return image, segmap


class Znormalization:
    def __call__(self, sample):
        image, segmap = sample
        image = (image - image.mean()) / image.std()
        return image, segmap


def get_transforms(data_config):
    val_transforms = [
        augmentations.random_clip(-100, 400),
    ]

    val_transforms += [
        Znormalization(),
        ToTensor(),
        augmentations.Resize(data_config.resize)
    ]

    train_transforms = [SliceVolume(slice_size=data_config.slice_size, force_non_empty=data_config.force_non_empty)]
    if data_config.augment_data:
        if data_config.elastic_deformations:
            train_transforms += [augmentations.ElasticDeformation3D(sigma=7, p=0.5)]

        train_transforms += [
            augmentations.RandomCrop(p=0.75),
            augmentations.random_clip((-200, -50), (256, 1024))
        ]

        train_transforms += [
            Znormalization(),
            ToTensor(),
            augmentations.random_flips(p=0.75),
            augmentations.RandomAffine(p=0.75, degrees=(-45, 45), translate=(0, 0.15), scale=(0.75, 1)),
            augmentations.Resize(data_config.resize),
            augmentations.random_noise(p=0.5, std_factor=0.25),
        ]
    else:
        train_transforms += val_transforms

    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def get_datasets(data_config):
    """
    Gather and split data paths and create datasets with according transforms
    param: split_mode: float for random split by percentage if validation cases or list of case numbers for the validation set
    """
    data_paths = get_data_pathes(data_config.data_path)
    random.shuffle(data_paths)

    train_paths = []
    val_paths = []
    for ct_path, gt_path in data_paths:
        is_train = True
        for case_num in LITS2017_VALSETS[data_config.val_set]:
            if f"volume-{case_num}" in ct_path:
                is_train = False
                break
        if is_train:
            train_paths.append((ct_path, gt_path))
        else:
            val_paths.append((ct_path, gt_path))

    train_transforms, val_transforms = get_transforms(data_config)
    tarin_set = CTDataset(train_paths, data_mode=data_config.data_mode,  transforms=train_transforms, delete_bakground=data_config.delete_background, ignore_background=data_config.ignore_background)
    val_set = CTDataset(val_paths, data_mode=data_config.data_mode, transforms=val_transforms, delete_bakground=data_config.delete_background, ignore_background=data_config.ignore_background)

    return tarin_set, val_set


def get_dataloaders(data_config):
    """
    Get dataloaders for training and evaluation.
    """
    train_set, val_set = get_datasets(data_config)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=data_config.batch_size, num_workers=data_config.num_workers)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=1, num_workers=data_config.num_workers)

    return train_loader, val_loader


class CTDataset(Dataset):
    """
    Dataset of entire CT volumes.
    """

    def __init__(self, data_paths, data_mode, transforms=None, delete_bakground=False, ignore_background=False):
        self.data_mode = data_mode
        self.transforms = transforms
        self.delete_bakground = delete_bakground
        self.ignore_background = ignore_background
        self.cts = []
        self.segs = []
        self.case_names = []
        n_slices = []
        for ct_path, seg_path in data_paths:
            self.segs.append(read_volume(seg_path).astype(np.uint8))
            self.cts.append(read_volume(ct_path))
            self.case_names.append(os.path.splitext(os.path.basename(ct_path))[0])
            n_slices.append(self.cts[-1].shape[-3])

        self.n_slices = np.sum(n_slices)
        logging.info(f"Dataset loaded: {self.n_slices} slices in {len(self.cts)} volumes")

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, i):
        sample = self.cts[i],  self.segs[i]
        if self.transforms:
            sample = self.transforms(sample)

        if self.data_mode == 'tumor':
            gt = (sample[1] == 2).long()
            mask = (sample[1] != 0)
        elif self.data_mode == 'liver':
            gt = (sample[1] != 0).long()
            mask = torch.ones_like(gt).bool()
        elif self.data_mode == 'multiclass':
            gt = sample[1]
            mask = torch.ones_like(gt).bool()
        else:
            raise ValueError("No such data mode: choose between 'tumor'/'liver'/'multiclass' ")

        if self.delete_bakground:
            sample[0][~mask] = (sample[0][~mask]).float().mean().to(dtype=sample[0].dtype)
        if not self.ignore_background:
            mask = torch.ones_like(mask).bool()

        return {'ct':  sample[0], "gt":  gt, 'mask': mask, 'case_name': self.case_names[i]}


if __name__ == '__main__':
    seg_dir = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/train/seg'
    import os
    for fname in os.listdir(seg_dir):
        x = read_volume(os.path.join(seg_dir, fname))
        if not np.any(x == 2):
            print(fname)




