import os
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import SimpleITK as sitk
from  scipy.ndimage.morphology import binary_dilation
import cc3d

IMG_MIN_VAL = -512
IMG_MAX_VAL = 512


def create_normal_dataset(root_dir, spatial_subsample, slice_size_mm=2, expand_slice=20, min_depth=48):
    """Downsample spacialy and normalize to have same number of equaly spaced slices"""

    processed_dir = f"Full-Torso-({spatial_subsample},{slice_size_mm})"
    new_ct_path = os.path.join(processed_dir, 'ct')
    new_seg_dir = os.path.join(processed_dir, 'seg')

    os.makedirs(new_ct_path, exist_ok=True)
    os.makedirs(new_seg_dir, exist_ok=True)

    for ct_filename in tqdm(os.listdir(os.path.join(root_dir, 'ct'))):
        # Read data
        ct = sitk.ReadImage(os.path.join(root_dir, 'ct', ct_filename), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(os.path.join(root_dir, 'seg', ct_filename.replace('volume', 'segmentation')),
                             sitk.sitkInt16)
        seg_array = sitk.GetArrayFromImage(seg)
        assert (seg_array.shape == ct_array.shape)

        # Manipulate values
        ct_array = np.clip(ct_array, IMG_MIN_VAL, IMG_MAX_VAL)
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
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / spatial_subsample),
                           ct.GetSpacing()[1] * int(1 / spatial_subsample), slice_size_mm))

        new_seg = sitk.GetImageFromArray(seg_array)

        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], slice_size_mm))

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, ct_filename))
        sitk.WriteImage(new_seg, os.path.join(new_seg_dir,
                                              ct_filename.replace('volume', 'segmentation').replace('.nii', '.nii')))


def crop_to_boxes_of_interset_cc(image_volume, labels_volume, relevant_lable, margins, allowed_perc_other_blobs, mask_dilation=11):
    crops = []

    binary_volume = labels_volume == relevant_lable
    binary_dilation(binary_volume, iterations=mask_dilation)

    cc = cc3d.connected_components(binary_volume)
    for label, image in cc3d.each(cc, binary=True, in_place=True):
        nwhere = np.where(image)
        relevant_ranges = tuple(slice(x.min(), x.max() + 1) for x in nwhere)
        values, counts = np.unique(cc[relevant_ranges].flatten(), return_counts=True)
        label_idx = np.where(values == label)[0][0]
        # label_occurances = counts[label_idx]
        other_non_bg_occurances = np.delete(counts, [0, label_idx]).sum()
        if other_non_bg_occurances / cc[relevant_ranges].size > allowed_perc_other_blobs:
            continue

        integer_margins = [margins[i] if i == 0 else int(margins[i] * (nwhere[i].max() - nwhere[i].min())) for i in range(3)]
        ranges_with_margins = tuple(slice(max(0, x.min()-integer_margins[i]), x.max()+integer_margins[i]) for i, x in enumerate(nwhere))
        crops.append((image_volume[ranges_with_margins], labels_volume[ranges_with_margins]))

    return crops


def create_tumor_dataset(root_dir, relevant_label, slice_size_mm=2, min_sizes=(4,10,10), slice_margins=(2, 0.75, 0.75), allowed_perc_other_blobs=0.5):
    """"
    Create a dataset of 3d crops of tumors with margins
    param: relevant_label: the lable value to regard as a binary value
    param: slice_size_mm: down/up sample in z dimension to normalize the real world size between CT slices to number of mm
    param: min_depth: minimal size in each dimension
    param: slice_margins: number padding units (slices/pixels) around area of interest to crop
    param: allowed_perc_other_blobs: drop samples containing more than this amount of voxel of non-main blob
    """
    processed_dir = f"Cropped_Tumoers_Dataset-(L-{relevant_label}_mm-{slice_size_mm})"
    new_ct_dir = os.path.join(processed_dir, 'ct')
    new_seg_dir = os.path.join(processed_dir, 'seg')

    os.makedirs(new_ct_dir, exist_ok=True)
    os.makedirs(new_seg_dir, exist_ok=True)

    for ct_filename in tqdm(os.listdir(os.path.join(root_dir, 'ct'))):
        # Read data
        ct = sitk.ReadImage(os.path.join(root_dir, 'ct', ct_filename), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(os.path.join(root_dir, 'seg', ct_filename.replace('volume', 'segmentation')), sitk.sitkInt16)
        seg_array = sitk.GetArrayFromImage(seg)
        assert (seg_array.shape == ct_array.shape)

        # Manipulate values
        ct_array = np.clip(ct_array, IMG_MIN_VAL, IMG_MAX_VAL)

        # Crop blobs and save with the same
        all_blobs = crop_to_boxes_of_interset_cc(ct_array, seg_array, relevant_label, slice_margins, allowed_perc_other_blobs)
        for blob_idx, (ct_array, seg_array) in enumerate(all_blobs):
            # seg_array[seg_array > 0] = 1

            # Resample
            new_dims = (ct.GetSpacing()[-1] / slice_size_mm, 1, 1)
            ct_array = ndimage.zoom(ct_array, new_dims, order=3)
            seg_array = ndimage.zoom(seg_array, new_dims, order=0)

            if np.any(ct_array.shape < np.array(min_sizes)):
                continue

            # Finally save data
            path = f"{os.path.join(new_ct_dir, os.path.splitext(ct_filename)[0])}-{blob_idx}.npy"
            np.save(path, ct_array)
            path = f"{os.path.join(new_seg_dir, os.path.splitext(ct_filename)[0])}-{blob_idx}.npy".replace(f'volume',f'segmentation')
            np.save(path, seg_array)


if __name__ == '__main__':
    raw_data = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/raw_data'
    # create_tumor_dataset(raw_data, relevant_label=2)
    create_normal_dataset(raw_data, spatial_subsample=0.5, slice_size_mm=2, expand_slice=5, min_depth=16)
    # create_tumor_dataset(raw_data, relevant_label=1, slice_margins=[1, 0, 0], allowed_perc_other_blobs=1)
