import os

import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from datasets.augmentations import *
from datasets.ct_dataset import CTDataset, get_data_pathes, Znormalization
from datasets.ct_dataset import ToTensor
from torchvision.utils import save_image
from torchvision import transforms as tv_transforms

# directions = [(-3, 'axial'), (-2, 'coronal'), (-1, 'sagittal')]
COLORS = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]


def get_3c_grayscale(volume):
    # Scale to values between 0 and 1
    mxval = torch.max(volume)
    mnval = torch.min(volume)
    im_volume = (volume - mnval) / max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255 * im_volume
    return torch.stack((im_volume, im_volume, im_volume), dim=-1)


def class_to_color(segmentation, colors):
    # initialize output to zeros
    seg_color = torch.zeros(segmentation.shape + (3,), device=segmentation.device)

    # set output to appropriate color at each location
    for i, c in enumerate(colors):
        if i > 0:
            seg_color[segmentation == i] = torch.tensor(c, dtype=seg_color.dtype, device=seg_color.device)
    return seg_color


def overlay(ct_volume, label_volume, alpha=0.3):
    # ct_volume.shape == label_volume.shape ==  (slices, h, w)
    # Get binary array for places where an ROI lives
    overlayed = get_3c_grayscale(ct_volume)

    if label_volume is not None:
        label_color_volume = class_to_color(label_volume, COLORS)
        segbin = torch.greater(label_volume, 0)
        repeated_segbin = torch.stack((segbin, segbin, segbin), dim=-1)
        # Weighted sum where there's a value to overlay
        overlayed = torch.where(
            repeated_segbin,
            alpha * label_color_volume + (1 - alpha) * overlayed,
            overlayed
        )
    overlayed = overlayed.permute(0, 3, 1, 2).to(torch.uint8)
    return overlayed


def write_volume_slices(ct_volume, additional_volumes, dir_path):
    """
    param ct_volume: tensor of shape (slices, H, W) read as ct volume
    param additional_volumes: list of integer tensor of shape (slices, H, W) read as label maps
    """
    os.makedirs(dir_path, exist_ok=True)
    for s in range(ct_volume.shape[0]):
        image_strip = [overlay(ct_volume[s].unsqueeze(0), None)]
        for label_volume in additional_volumes:
            image_strip.append(overlay(ct_volume[s].unsqueeze(0), label_volume[s].unsqueeze(0)))
        image_strip = torch.cat(image_strip, dim=-1)
        save_image(image_strip.float(), f"{dir_path}/slice-{s}.png", normalize=True)


def visualize_augmentations(data_paths, outputs_dir):
    """
    Visualize the isolated effect of each image augmentation
    """
    transforms = [
        (ToTensor(), 1, "original-clipped"),
        (tv_transforms.Compose([
            ElasticDeformation3D(sigma=7, p=0.5),
            RandomCrop(p=1),
            random_clip((-200, -50), (256, 1024)),
            Znormalization(),
            ToTensor(),
            random_flips(p=1),
            RandomAffine(p=0.75, degrees=(-45, 45), translate=(0, 0.15), scale=(0.75, 1)),
            Resize(128),
            random_noise(p=0.5, std_factor=0.25),
        ]), 10, 'train_transforms')
    ]

    for (t, n_repeats, t_name) in transforms:
        dataloader = DataLoader(CTDataset(data_paths, transforms=t))
        for i in range(n_repeats):
            visualize_dataset(dataloader, os.path.join(outputs_dir, f"{t_name}-{i}"))

        for sample in dataloader:
            case_name = sample['case_name'][0]
            create_augmentation_gif(os.path.join(outputs_dir, f"original-clipped-{0}", case_name),
                                    [os.path.join(outputs_dir, f"{t_name}-{i}", case_name) for i in range(n_repeats)],
                                    63,
                                    os.path.join(outputs_dir, f"{t_name}_{case_name}-{63}.gif"))


def create_augmentation_gif(original_slices_dir, transformed_slices_dirs, slice_number, save_path):
    import imageio
    path = os.path.join(original_slices_dir, f"slice-{slice_number}.png")
    if os.path.exists(path):
        images = [imageio.imread(path)]
        for t_dir in transformed_slices_dirs:
            images.append(imageio.imread(os.path.join(t_dir, f"slice-{slice_number}.png")))
        imageio.mimsave(save_path, images, duration=1)


def visualize_dataset(dataloader, output_dir):
    """
    Writes slices of all volumes in dataloader with their GTs
    """
    os.makedirs(output_dir, exist_ok=True)

    # iterate over the validation set
    for sample in dataloader:
        ct_volume = sample['ct']
        gt_volume = sample['gt']
        mask_volume = sample['mask']
        assert (ct_volume.shape[0] == 1)
        volume_dir = f"{output_dir}/{sample['case_name'][0]}"

        write_volume_slices(ct_volume[0], [gt_volume[0], mask_volume[0]], volume_dir)


if __name__ == '__main__':
    # visualize original data and its preprocessing
    # original_data_path = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/train'
    original_data_path = 'LiTS2017C-(3, 10, 10)'
    data_paths = get_data_pathes(original_data_path)
    sorted(data_paths)
    data_paths = [x for x in data_paths if "46" in x[0]]

    dataloader = DataLoader(CTDataset(data_paths, transforms=torchvision.transforms.Compose([random_clip(-100, 400), ToTensor()])))
    visualize_dataset(dataloader, os.path.join(original_data_path, "visualize_data"))
    exit()
    # visualize_preprocessing_affects(dataloader, os.path.join(original_data_path, "visualize_preprocessing"))

    # # # Viuslize dataset
    # data_path = '/mnt/storage_ssd/train_outputs/cluster_LiverTraining/VGGUNet_Aug_Loss(0.0Dice+0.0WCE+1.0CE)_V-A'
    # data_paths = get_data_pathes(data_path)
    # sorted(data_paths)
    # data_paths = data_paths
    # #
    # dataloader = DataLoader(CTDataset(data_paths, transforms=ToTensor()))
    # visualize_dataset(dataloader, os.path.join(data_path, "visualize_data"))
    #
    # visualize_augmentations(data_paths, os.path.join(data_path, "visualize_augmentations"))
    #
    # import torchio as tio
    # path = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/test/ct/test-volume-4.nii'
    # x = tio.ScalarImage(path)
    # x = tio.Clamp(-100,400)(x).plot(axis=False)