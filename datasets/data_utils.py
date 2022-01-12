import SimpleITK as sitk
import os
import numpy as np


def read_volume(path):
    if path.endswith("nii"):
        volume = sitk.ReadImage(path, sitk.sitkInt16)
        volume = sitk.GetArrayFromImage(volume)
    else:
        volume = np.load(path)
    return volume


def get_data_pathes(data_root):
    ct_files = sorted(os.listdir(os.path.join(data_root, 'ct')))
    segmentation_files = sorted(os.listdir(os.path.join(data_root, 'seg')))

    ct_files = [os.path.join(data_root, 'ct', x) for x in ct_files]
    segmentation_files = [os.path.join(data_root, 'seg', x) for x in segmentation_files]

    data_paths = list(zip(ct_files, segmentation_files))

    return data_paths