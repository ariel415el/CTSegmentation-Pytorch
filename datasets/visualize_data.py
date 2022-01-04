import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import augmentations
from datasets.ct_dataset import CTDataset
from utils import overlay


def visualize_dataset(data_path):
    output_dir = os.path.join(data_path, "visulization")
    os.makedirs(output_dir, exist_ok=True)

    ct_files = sorted(os.listdir(os.path.join(data_path, 'ct')))
    ct_files = [os.path.join(data_path, 'ct', x) for x in ct_files]
    segmentation_files = sorted(os.listdir(os.path.join(data_path, 'seg')))
    segmentation_files = [os.path.join(data_path, 'seg', x) for x in segmentation_files]

    data_paths = list(zip(ct_files, segmentation_files))

    dataloader = DataLoader(CTDataset(data_paths, transforms=augmentations.ToTensor()))

    # iterate over the validation set
    for b_idx, (ct_volume, gt_volume) in enumerate(dataloader):
        volume_dir = f"{output_dir}/Volume-{b_idx}"
        os.makedirs(volume_dir, exist_ok=True)

        for i in range(ct_volume.shape[0]):
            img1 = overlay(ct_volume[i], gt_volume[i])
            img2 = overlay(ct_volume[i], gt_volume[i]*0)
            img = torch.cat([img1, img2], dim=-1)
            for s in range(ct_volume.shape[-3]):
                save_image(img[s], f"{volume_dir}/{i}-{s}.png", normalize=True)


if __name__ == '__main__':
    visualize_dataset("Cropped_Tumoers_Dataset-(L-2_mm-2)")