import numpy as np
import torch
from torchvision.transforms import transforms, InterpolationMode
from scipy.ndimage import map_coordinates
import torchvision.transforms.functional as F


class Resize:
    def __init__(self, rescale=128):
        self.rescale = rescale
        self.resize_ct = transforms.Resize((self.rescale, self.rescale))
        self.resize_gt = transforms.Resize((self.rescale, self.rescale), interpolation=InterpolationMode.NEAREST)

    def __call__(self, sample):
        image, segmap = sample

        image = self.resize_ct(image)
        segmap = self.resize_gt(segmap)

        # image = transforms.Resize((self.rescale, self.rescale))(torch.from_numpy(image).unsqueeze(0))[0].numpy()
        # segmap = transforms.Resize((self.rescale, self.rescale), interpolation=InterpolationMode.NEAREST)(segmap[None, :])[0]

        return image, segmap


class HistogramEqualization:
    def __init__(self, nbins):
        self.nbins = nbins

    def __call__(self, sample):
        image, segmap = sample
        image_histogram, bins = np.histogram(image.flatten(), bins=256, density=True)
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = cdf / cdf[-1]  # normalize to [0,255]

        # use linear interpolation of cdf to find new pixel values
        image = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)

        

        return image, segmap



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

        return image, segmap


class RandomCrop:
    """Crop randomly the image in a sample.
    Args:
        scale_range: tuple: range of possible crop factor for each dimenstion.
    """

    def __init__(self, p=0.5, scale_range=(0.8, 1)):
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


# class ElasticDeformation3D_package:
#     def __init__(self, sigma=25, n_points=3, p=0.5):
#         self.sigma = sigma
#         self.n_points = n_points
#         self.p = p
#
#     def __call__(self, sample):
#         if torch.rand(1) < self.p:
#             import elasticdeform
#             # return elasticdeform.deform_random_grid([X, Y], sigma=self.sigma, order=[1, 0], mode='nearest', axis=(1, 2)) # only spatialy (same for all slices)
#             sample = elasticdeform.deform_random_grid(list(sample), sigma=self.sigma, order=[1, 0], mode='nearest')
#         return image, segmap


class ElasticDeformation3D:
    def __init__(self, sigma=25, n_points=3, p=0.5, order=1):
        """
        taken from https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py
        Elastic deformation of 2D or 3D images on a gridwise basis
        X: image
        Y: segmentation of the image
        sigma = standard deviation of the normal distribution
        points = number of points of the each side of the square grid
        Elastic deformation approach found in
            Ronneberger, Fischer, and Brox, "U-Net: Convolutional Networks for Biomedical
            Image Segmentation" also used in Çiçek et al., "3D U-Net: Learning Dense Volumetric
            Segmentation from Sparse Annotation"
        based on a coarsed displacement grid interpolated to generate displacement for every pixel
        deemed to represent more realistic, biologically explainable deformation of the image
        for each dimension, a value for the displacement is generated on each point of the grid
        then interpolated to give an array of displacement values, which is then added to the corresponding array of coordinates
        the resulting (list of) array of coordinates is mapped to the original image to give the final image
        """
        self.sigma = sigma
        self.n_points = n_points
        self.order=order
        self.p = p

    def __call__(self, sample):
        image, segmap = sample
        if torch.rand(1) < self.p:
            S, H, W = image.shape

            # creates the grid of coordinates of the voxels of the image (an ndim array per dimension)
            voxel_coordinates = np.meshgrid(np.arange(S),
                                            np.arange(H),
                                            np.arange(W),
                                            indexing='ij')

            # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
            coordinate_grid_0_to_n_points = np.meshgrid(np.linspace(0, self.n_points - 1, S),
                                                        np.linspace(0, self.n_points - 1, H),
                                                        np.linspace(0, self.n_points - 1, W),
                                                        indexing='ij')

            # creates the deformation along each dimension and then add it to the coordinates
            for i in range(len(voxel_coordinates)):
                rand_displacements = np.random.randn(self.n_points, self.n_points, self.n_points) * self.sigma  # creating the displacement at the control points
                interp_displacements = map_coordinates(rand_displacements, coordinate_grid_0_to_n_points, order=self.order).reshape(image.shape)
                voxel_coordinates[i] = np.add(voxel_coordinates[i], interp_displacements)  # adding the displacement

            image = map_coordinates(image, voxel_coordinates, order=self.order, mode='nearest').reshape(image.shape)
            segmap = map_coordinates(segmap, voxel_coordinates, order=0, mode='nearest').reshape(segmap.shape)

        return image, segmap


class RandomAffine:
    def __init__(self, p=0.5, degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)):
        self.p = p
        self.degrees = list(degrees if degrees is not None else [0,0])
        self.translate = list(translate if translate is not None else [0,0])
        self.scale = list(scale if scale is not None else [1,1])

    def __call__(self, sample):
        image, segmap = sample
        if torch.rand(1) < self.p:
            image_size = F._get_image_size(image)
            ret = transforms.RandomAffine.get_params(self.degrees, self.translate, self.scale, None, img_size=image_size)
            image = F.affine(image, *ret, interpolation=InterpolationMode.BILINEAR)
            segmap = F.affine(segmap, *ret, interpolation=InterpolationMode.NEAREST)

        return image, segmap


class random_flips:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, segmap = sample
        if torch.rand(1) < self.p:
            if torch.rand(1) < 0.5:
                image, segmap = F.hflip(image), F.hflip(segmap)
            if torch.rand(1) < 0.5:
                image, segmap = F.vflip(image), F.vflip(segmap)

        return image, segmap


class random_noise:
    def __init__(self, p=0.5, std_factor=0.5):
        self.p = p
        self.std_factor = std_factor

    def __call__(self, sample):
        image, segmap = sample
        if torch.rand(1) < self.p:
            dtype = image.dtype
            image = image.float()
            image += torch.randn(image.shape) * self.std_factor * image.std()
            image = image.to(dtype=dtype)

        return image, segmap


class random_clip:
    def __init__(self, min_interval=(-512, -511), max_interval=(512,513)):
        self.min_interval = min_interval
        self.max_interval = max_interval

    def __call__(self, sample):
        image, segmap = sample

        min_v = self.min_interval if type(self.min_interval) == int else np.random.randint(*self.min_interval)
        max_v = self.max_interval if type(self.max_interval) == int else np.random.randint(*self.max_interval)
        image = image.clip(min_v, max_v)


        return image, segmap
