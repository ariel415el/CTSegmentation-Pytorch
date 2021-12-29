import os
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage


def process_data(root_dir, spatial_subsample, slice_size_mm=2, img_min_val=-200, img_max_val=200, expand_slice=20, min_depth=48):
    """Downsample spacialy and normalize to have same number of equaly spaced slices"""

    processed_dir = f"processed-data-({spatial_subsample},{slice_size_mm})"
    new_ct_path = os.path.join(processed_dir, 'ct')
    new_seg_dir = os.path.join(processed_dir, 'seg')

    os.makedirs(new_ct_path, exist_ok=True)
    os.makedirs(new_seg_dir, exist_ok=True)

    for ct_filename in tqdm(os.listdir(os.path.join(root_dir, 'ct'))):
        # Read data
        ct = sitk.ReadImage(os.path.join(root_dir, 'ct', ct_filename), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(os.path.join(root_dir, 'seg', ct_filename.replace('volume', 'segmentation')), sitk.sitkInt16)
        seg_array = sitk.GetArrayFromImage(seg)
        assert(seg_array.shape == ct_array.shape)

        # Manipulate values
        ct_array = np.clip(ct_array, img_min_val, img_max_val)
        # seg_array[seg_array > 0] = 1

        # Resample
        new_dims = (ct.GetSpacing()[-1] / slice_size_mm, spatial_subsample, spatial_subsample)
        ct_array = ndimage.zoom(ct_array, new_dims, order=3)
        seg_array = ndimage.zoom(seg_array, new_dims, order=0)

        # Restrict data to interesteing parts only
        z = np.any(seg_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        start_slice = max(0, start_slice - expand_slice)
        end_slice = min(seg_array.shape[0] - 1, end_slice + expand_slice)

        if end_slice - start_slice + 1 < min_depth:
            continue

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        # Finally save data as NII
        new_ct = sitk.GetImageFromArray(ct_array)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / spatial_subsample), ct.GetSpacing()[1] * int(1 / spatial_subsample), slice_size_mm))

        new_seg = sitk.GetImageFromArray(seg_array)

        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], slice_size_mm))

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, ct_filename))
        sitk.WriteImage(new_seg, os.path.join(new_seg_dir, ct_filename.replace('volume', 'segmentation').replace('.nii', '.nii')))


if __name__ == '__main__':
    process_data('/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/raw_data', 0.5)