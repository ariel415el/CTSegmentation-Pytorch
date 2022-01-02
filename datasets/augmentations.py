import cv2
import numpy as np
import torch
from torchvision.transforms import transforms, InterpolationMode
from skimage.transform import resize


class RandomScale:
    def __init__(self, p=0.5, scale_range=(128,256)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, sample):
        image, segmap = sample
        if torch.rand(1) < self.p:
            h, w = image.shape[-2:]
            new_w = np.random.randint(*self.scale_range)
            new_h = h * new_w // w

            image = transforms.Resize((new_h, new_w))(torch.from_numpy(image).unsqueeze(0))[0].numpy()
            segmap = transforms.Resize((new_h, new_w), interpolation=InterpolationMode.NEAREST)(torch.from_numpy(segmap).unsqueeze(0))[0].numpy()
            # image = cv2.resize(segmap, (new_w, new_h))
            # segmap = cv2.resize(segmap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # image = resize(image, (new_h, new_w), order=1)
            # segmap = resize(segmap, (new_h, new_w), order=0)

        return image, segmap


class Resize:
    def __init__(self, rescale=128):
        self.rescale = rescale

    def __call__(self, sample):
        image, segmap = sample

        image = transforms.Resize((self.rescale, self.rescale))(torch.from_numpy(image).unsqueeze(0))[0].numpy()
        segmap = transforms.Resize((self.rescale, self.rescale), interpolation=InterpolationMode.NEAREST)(torch.from_numpy(segmap).unsqueeze(0))[0].numpy()

        return image, segmap


class RandomCrop:
    """Crop randomly the image in a sample.
    Args:
        scale_range: tuple: range of possible crop factor for each dimenstion.
    """

    def __init__(self, p=0.5, scale_range=(0.8,1)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, sample):
        image, segmap = sample
        if torch.rand(1) < self.p:
            h, w = image.shape[-2:]
            new_h = int(np.random.uniform(*self.scale_range) * h)
            new_w = int(np.random.uniform(*self.scale_range) * w)

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[..., top: top + new_h, left: left + new_w]
            segmap = segmap[..., top: top + new_h, left: left + new_w]

        return image, segmap


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