import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import augmentations
from datasets.ct_dataset import CTDataset, get_data_pathes, read_volume
from utils import overlay


def visualize_dataset(data_path):
    output_dir = os.path.join(data_path, "visulization")
    os.makedirs(output_dir, exist_ok=True)

    data_paths = get_data_pathes(data_path)
    dataloader = DataLoader(CTDataset(data_paths, transforms=augmentations.ToTensor()))

    # dataloader = [{'ct': torch.from_numpy(read_volume(f'{data_path}/ct/volume-41.nii').clip(-512,512)[None, :]),
    #               'gt': torch.from_numpy(read_volume(f'{data_path}/seg/segmentation-41.nii')[None, :]),
    #                'case_name':['volume-41']}]

    # iterate over the validation set
    for sample in dataloader:
        ct_volume = sample['ct']
        gt_volume = sample['gt']
        assert(ct_volume.shape[0] == 1)
        volume_dir = f"{output_dir}/{sample['case_name'][0]}"
        os.makedirs(volume_dir, exist_ok=True)

        img1 = overlay(ct_volume[0], gt_volume[0])
        img2 = overlay(ct_volume[0], gt_volume[0]*0)
        img = torch.cat([img1, img2], dim=-1)
        for s in range(ct_volume.shape[-3]):
            save_image(img[s], f"{volume_dir}/slice-{s}.png", normalize=True)


if __name__ == '__main__':
    visualize_dataset('/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/raw_data')
    # visualize_dataset('LiverData_(S-1_MS-(3, 30, 30)RL-True_CP-[CL-1_margins-(1, 1, 1)_OB-0.5_MD-7])')
