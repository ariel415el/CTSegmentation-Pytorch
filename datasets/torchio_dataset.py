import numpy as np
import torchio as tio
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import nibabel as nib
from torchvision import transforms
from data_utils import get_data_pathes

class RandomSliceVolume(object):
    """Slice a portion of fixed size from the volume
      return a slice of size "slice_size" in the -4 dimension of the sample.
    """
    def __init__(self, slice_size=48):
        self.slice_size = slice_size

    def __call__(self,  image):
        if image.shape[-1] > self.slice_size:
            start_slice = np.random.randint(0, image.shape[-1] - self.slice_size)
            end_slice = start_slice + self.slice_size - 1

            image = image[..., start_slice:end_slice + 1]

        return image


def tio_to_tensor(subject):
    return {'ct': subject.ct.tensor.permute(0, 3, 1, 2)[0],
            'gt': subject.gt.tensor.permute(0, 3, 1, 2)[0],
            'case_name': subject.case_name}


def get_transforms(slice_size, resize):
    train_transforms = [
        tio.CropOrPad(mask_name='gt'),
        tio.ToCanonical(),
        tio.RandomAnisotropy(p=0.25),                            # resample axial dimension (could be faster after crio)
        tio.Lambda(RandomSliceVolume(slice_size=slice_size)),
        tio.CropOrPad((resize, resize, slice_size)),
        tio.RescaleIntensity((-1, 1)),
        tio.RandomBlur(p=0.25),  # blur 25% of times
        tio.RandomNoise(p=0.25),  # Gaussian noise 25% of times
        # tio.OneOf({  # either
        #     tio.RandomAffine(): 0.7,  # random affine
        #     tio.RandomElasticDeformation(): 0.3,  # or random elastic deformation
        # }, p=0.8),  # applied to 80% of images
        tio_to_tensor
    ]

    val_transforms = [
        # augmentations.Resize(resize),
    ]
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


def get_datasets(data_root, val_perc, slice_size, resize):
    data_paths = get_data_pathes(data_root)
    subjects = [
        tio.Subject(ct=tio.ScalarImage(ct_path),
                    gt=tio.LabelMap(gt_path),
                    case_name=os.path.splitext(os.path.basename(ct_path))[0])
        for (ct_path, gt_path) in data_paths
    ]

    train_transforms, val_transforms = get_transforms(slice_size, resize)
    n_val = int(len(data_paths) * val_perc)
    tarin_set = tio.SubjectsDataset(subjects[n_val:], transform=train_transforms)
    val_set = tio.SubjectsDataset(subjects[:n_val], transform=val_transforms)
    return tarin_set, val_set


def get_dataloaders(data_root, val_perc, params, slice_size=1, resize=256):
    """
    Get dataloaders for training and evaluation.
    train_by_volume: 3d/2d training returns full CT volumes (batch_size, slices, H, W) or (batch_size, H, W)
    """
    train_set, val_set = get_datasets(data_root, val_perc, slice_size, resize)

    train_loader = DataLoader(train_set, shuffle=True, **params)
    params['batch_size'] = 1
    val_loader = DataLoader(val_set, shuffle=True, **params)

    return train_loader, val_loader


# @title Visualization functions
def get_bounds(self):
    """Get image bounds in mm.

    Returns:
        np.ndarray: [description]
    """
    first_index = 3 * (-0.5,)
    last_index = np.array(self.spatial_shape) - 0.5
    first_point = nib.affines.apply_affine(self.affine, first_index)
    last_point = nib.affines.apply_affine(self.affine, last_index)
    array = np.array((first_point, last_point))
    bounds_x, bounds_y, bounds_z = array.T.tolist()
    return bounds_x, bounds_y, bounds_z


def to_pil(image):
    from PIL import Image
    from IPython.display import display
    data = image.numpy().squeeze().T
    data = data.astype(np.uint8)
    image = Image.fromarray(data)
    w, h = image.size
    display(image)
    print()  # in case multiple images are being displayed


def stretch(img):
    p1, p99 = np.percentile(img, (1, 99))
    from skimage import exposure
    img_rescale = exposure.rescale_intensity(img, in_range=(p1, p99))
    return img_rescale


def show_subject_slice(subject, stretch_slices=True, parcellation=True):
    print(subject.ct.data.shape)
    subject = tio.ToCanonical()(subject)

    def flip(x):
        return np.rot90(x)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    half_shape = torch.Tensor(subject.spatial_shape) // 2
    i, j, k = half_shape.long()
    # i -= 5  # use a better slice

    bounds_x, bounds_y, bounds_z = get_bounds(subject.ct)  ###

    orientation = ''.join(subject.ct.orientation)
    if orientation != 'RAS':
        import warnings
        warnings.warn(f'Image orientation should be RAS+, not {orientation}+')

    kwargs = dict(cmap='gray', interpolation='none')
    data = subject['ct'].data
    slices = data[0, i], data[0, :, j], data[0, ..., k]
    if stretch_slices:
        slices = [stretch(s.numpy()) for s in slices]
    sag, cor, axi = slices

    axes[0, 0].imshow(flip(sag), extent=bounds_y + bounds_z, **kwargs)
    axes[0, 1].imshow(flip(cor), extent=bounds_x + bounds_z, **kwargs)
    axes[0, 2].imshow(flip(axi), extent=bounds_x + bounds_y, **kwargs)

    kwargs = dict(interpolation='none')
    data = subject.gt.data
    slices = data[0, i], data[0, :, j], data[0, ..., k]
    if parcellation:
        sag, cor, axi = [colorize(s.long()) for s in slices]
    else:
        sag, cor, axi = slices
    axes[1, 0].imshow(flip(sag), extent=bounds_y + bounds_z, **kwargs)
    axes[1, 1].imshow(flip(cor), extent=bounds_x + bounds_z, **kwargs)
    axes[1, 2].imshow(flip(axi), extent=bounds_x + bounds_y, **kwargs)

    plt.tight_layout()
    plt.show()

def colorize(label_map):
    rgb = np.stack(3 * [label_map], axis=-1)
    rgb[label_map == 1] = (0, 0, 255)
    rgb[label_map == 2] = (255, 0, 0)
    return rgb


if __name__ == '__main__':
    from torchvision.utils import save_image
    import torch
    from datasets.visualize_data import overlay
    import os

    outputs_dir = "visualize_augmentations"
    os.makedirs(outputs_dir, exist_ok=True)
    # data_path = 'datasets/LiverData_(S-1_MS-(3, 30, 30)RL-True_CL-1_margins-(1, 1, 1)_OB-0.5_MD-7)'
    # data_path = r'/home/ariel/projects/MedicalImageSegmentation/CTSegmentation-Pytorch/datasets/LiverData_(S-1_MS-(3, 30, 30)_RL-True_CP-CL-1_margins-(1, 1, 1)_OB-0.5_MD-7)'
    data_path = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/raw_data'
    params = dict(batch_size=1, num_workers=0)
    train_loader, _ = get_dataloaders(data_path, val_perc=0.1, params=params, slice_size=16, resize=400)

    for sample in train_loader:
        ct = sample['ct'][0]
        gt = sample['gt'][0]
        case_name = sample['case_name'][0]

        raw = overlay(ct, gt * 0)
        gt_vis = overlay(ct, gt)
        imgs = torch.cat([raw, gt_vis], dim=-1)
        for s in range(ct.shape[0]):
            save_path = os.path.join(outputs_dir, f"Case-{case_name}-slice-{s}_.png")
            save_image(imgs[s], save_path, normalize=True)
