import os

import torch
from torch.utils.data import DataLoader
from datasets.augmentations import *
from datasets.ct_dataset import CTDataset, get_data_pathes
from datasets.ct_dataset import ToTensor
from datasets.preprocess_data import manipulate_CT_intencities, intencity_parameterss
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
    overlayed = overlayed.permute(0, 3, 1, 2)
    return overlayed


def write_volume_slices(ct_volume, additional_volumes, dir_path):
    """
    param ct_volume: tensor of shape (slices, H, W) read as ct volume
    param additional_volumes: list of integer tensor of shape (slices, H, W) read as label maps
    """
    os.makedirs(dir_path, exist_ok=True)
    image_volumes = [overlay(ct_volume, None)]
    for label_volume in additional_volumes:
        image_volumes.append(overlay(ct_volume, label_volume))

    img_volume = torch.cat(image_volumes, dim=-1)
    for s in range(ct_volume.shape[-3]):
        save_image(img_volume[s], f"{dir_path}/slice-{s}.png", normalize=True)


def visualize_preprocessing_affects(dataloader, output_dir):
    param_sets = [
        intencity_parameterss(clip_values=None),
        intencity_parameterss(clip_values=(-100, 400)),
        intencity_parameterss(clip_values=(-512, 512)),
        intencity_parameterss(clip_values=None, hist_equalization=True),
        intencity_parameterss(clip_values=(-100, 400), hist_equalization=True),
        intencity_parameterss(clip_values=(-512, 512), hist_equalization=True),
    ]
    for sample in dataloader:
        for i, params in enumerate(param_sets):
            ct_volume = sample['ct']
            gt_volume = sample['gt']
            ct_volume = torch.from_numpy(manipulate_CT_intencities(ct_volume.numpy(), params))

            volume_dir = f"{output_dir}/{sample['case_name'][0]}/{str(params)}"
            write_volume_slices(ct_volume[0], [gt_volume[0]], volume_dir)


def visualize_augmentations(data_paths, outputs_dir):
    """
    Visualize the isolated effect of each image augmentation
    """

    transforms = [
        (tv_transforms.Compose([ToTensor()]), 1, 'raw'),
        (tv_transforms.Compose([random_clip(-100, 400), ToTensor()]), 1, 'original-clipped'),
        (tv_transforms.Compose([histogram_equalization(nbins=256), ToTensor()]), 1, 'HistEqual'),
        (tv_transforms.Compose([random_clip((-200, -50), (256, 1024)), ToTensor()]), 3, 'Rclip'),
        (tv_transforms.Compose([random_clip(-100, 400), ElasticDeformation3D(sigma=5, p=1), ToTensor()]), 3, 'elastic'),
        (tv_transforms.Compose([random_clip(-100, 400), RandomCrop(p=1), ToTensor()]), 3, 'Rcrop'),
        (tv_transforms.Compose([random_clip(-100, 400), ToTensor(), Resize(256)]), 1, 'Resize'),
        (tv_transforms.Compose([random_clip(-100, 400), ToTensor(), random_flips(p=1)]), 3, 'flip'),
        (tv_transforms.Compose([random_clip(-100, 400), ToTensor(), random_noise(p=1, std=0.25)]), 1, 'noise'),
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
    original_data_path = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/train'
    data_paths = get_data_pathes(original_data_path)
    sorted(data_paths)
    data_paths = data_paths[12:13]

    dataloader = DataLoader(CTDataset(data_paths, transforms=ToTensor()))
    visualize_dataset(dataloader, os.path.join(original_data_path, "visualize_data"))

    # visualize_preprocessing_affects(dataloader, os.path.join(original_data_path, "visualize_preprocessing"))

    # # Viuslize dataset
    # data_path = 'datasets/LiverData_(S-1_MS-(3, 10, 10)_Crop-CL-1_margins-(1, 1, 1)_OB-0.5_MD-11)'
    # data_paths = get_data_pathes(data_path)
    # sorted(data_paths)
    # data_paths = data_paths[32:33]
    #
    # visualize_augmentations(data_paths, os.path.join(data_path, "visualize_augmentations"))
