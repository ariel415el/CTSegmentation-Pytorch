import numpy as np
import torch
from torchvision.transforms import transforms, InterpolationMode
from scipy.ndimage import map_coordinates


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


class ElasticDeformation3D_package:
    def __init__(self, sigma=25, n_points=3, p=0.5):
        self.sigma = sigma
        self.n_points = n_points
        self.p = p

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            import elasticdeform
            # return elasticdeform.deform_random_grid([X, Y], sigma=self.sigma, order=[1, 0], mode='nearest', axis=(1, 2)) # only spatialy (same for all slices)
            return elasticdeform.deform_random_grid(list(sample), sigma=self.sigma, order=[1, 0], mode='nearest')


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
        if torch.rand(1) < self.p:
            X, Y = sample
            shape = X.shape

            # creates the grid of coordinates of the voxels of the image (an ndim array per dimension)
            voxel_coordinates = np.meshgrid(np.arange(shape[0]),
                                      np.arange(shape[1]),
                                      np.arange(shape[2]),
                                      indexing='ij')

            # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
            coordinate_grid_0_to_n_points = np.meshgrid(np.linspace(0, self.n_points - 1, shape[0]),
                             np.linspace(0, self.n_points - 1, shape[1]),
                             np.linspace(0, self.n_points - 1, shape[2]),
                             indexing='ij')

            # creates the deformation along each dimension and then add it to the coordinates
            for i in range(len(shape)):
                rand_displacements = np.random.randn(self.n_points, self.n_points, self.n_points) * self.sigma  # creating the displacement at the control points
                interp_displacements = map_coordinates(rand_displacements, coordinate_grid_0_to_n_points, order=self.order).reshape(shape)
                voxel_coordinates[i] = np.add(voxel_coordinates[i], interp_displacements)  # adding the displacement

            X = map_coordinates(X, voxel_coordinates, order=self.order, mode='nearest').reshape(shape)
            Y = map_coordinates(Y, voxel_coordinates, order=0, mode='nearest').reshape(shape)
            sample = X,Y

        return sample


# class tio_wrapper:
#     def __init__(self, tio_transform):
#         self.tio_transform = tio_transform
#
#     def __call__(self, sample):
#         x = {'ct': torch.from_numpy(sample[0][None, :]), 'gt': torch.from_numpy(sample[1][None, :])}
#         x = self.tio_transform(x)
#         return x['ct'][0].numpy(), x['gt'][0].numpy()


if __name__ == '__main__':
    x = np.ones((5,128,128))
    y = np.ones((5,128,128)).astype(int)

    trnsf = ElasticDeformation3D(sigma=1)
    x,y = trnsf((x,y))
    print(x.shape, y.shape)
