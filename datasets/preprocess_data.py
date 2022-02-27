import os
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import SimpleITK as sitk


def get_LiTS2017_paths(data_root):
    ct_files = sorted(os.listdir(os.path.join(data_root, 'ct')))
    segmentation_files = sorted(os.listdir(os.path.join(data_root, 'seg')))

    ct_files = [os.path.join(data_root, 'ct', x) for x in ct_files]
    segmentation_files = [os.path.join(data_root, 'seg', x) for x in segmentation_files]

    data_paths = list(zip(ct_files, segmentation_files))

    return data_paths


def get_KiTS2019_paths(data_root):
    return [
        (os.path.join(data_root, case, 'imaging.nii.gz'), os.path.join(data_root, case, 'segmentation.nii.gz'))
         for case in os.listdir(data_root)
    ]


def create_dataset(data_paths, crop_padding=(3,10,10), normalize_axial_mm=None, spatal_resize=1.0):
    """"
    Create a dataset of 3d crops of tumors with margins
    param: crop_params: optional parameters for cropped version of the data
    param: min_sizes: minimal dimensions for a volume
    param: spatial_scale: spatial scale factor
    #### param: slice_size_mm: down/up sample in z dimension to normalize the real world size between CT slices to number of mm

    """
    processed_dir = f"LiTS2017" + (f"C-{crop_padding}" if crop_padding is not None else "") + (f"{normalize_axial_mm}-mm" if normalize_axial_mm is not None else "")

    new_ct_dir = os.path.join(processed_dir, 'ct')
    new_seg_dir = os.path.join(processed_dir, 'seg')

    os.makedirs(new_ct_dir, exist_ok=True)
    os.makedirs(new_seg_dir, exist_ok=True)

    for ct_filepath, gt_fname in tqdm(data_paths):
        # Read data
        ct = sitk.ReadImage(ct_filepath, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(gt_fname, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        assert (seg_array.shape == ct_array.shape)

        if crop_padding is not None:
            # Crop around liver for tumor segmentaiton
            nwhere = np.where(seg_array != 0)
            print(nwhere)
            liver_crop = tuple([slice(max(0, x.min() - crop_padding[i]), x.max() + crop_padding[i]) for i, x in enumerate(nwhere)])
            ct_array = ct_array[liver_crop]
            seg_array = seg_array[liver_crop]

        # Resample
        if normalize_axial_mm is not None:
            new_dims = (ct.GetSpacing()[-1] / normalize_axial_mm, 1, 1)
            ct_array = ndimage.zoom(ct_array, new_dims, order=3)
            seg_array = ndimage.zoom(seg_array, new_dims, order=0)

        if spatal_resize < 1:
            new_dims = (1, spatal_resize, spatal_resize)
            ct_array = ndimage.zoom(ct_array, new_dims, order=3)
            seg_array = ndimage.zoom(seg_array, new_dims, order=0)

        # Finally save data
        fname = os.path.basename(ct_filepath)

        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetSpacing(ct.GetSpacing())
        sitk.WriteImage(new_ct, os.path.join(new_ct_dir, fname))

        new_seg = sitk.GetImageFromArray(seg_array.astype(np.uint8))
        new_seg.SetSpacing(ct.GetSpacing())
        sitk.WriteImage(new_seg, os.path.join(new_seg_dir, fname).replace(f'volume', f'segmentation'))

    print("Done")


if __name__ == '__main__':
    data_paths = get_LiTS2017_paths('/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/train')
    create_dataset(data_paths, crop_padding=(3,10,10))
