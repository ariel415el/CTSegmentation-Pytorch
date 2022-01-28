import os
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation
import cc3d

# @dataclass
class cropping_parameterss:
    """
    param: crop_by_label: create crops around 3d blobs of this values
    param: slice_margins: number padding units (slices/pixels) around area of interest to crop
    param: allowed_perc_other_blobs: drop samples containing more than this amount of voxel of non-main blob
    param: mask_dilation: how many dilation iterations to the binary image before finding blobs in it (more dilation = bigger allowed gap inside blobs)
    """
    def __init__(self, cropping_lable=1, slice_margins=(2, 10, 10), allowed_perc_other_blobs=0.5, mask_dilation=3):
        self.cropping_lable = cropping_lable
        self.slice_margins = slice_margins
        self.allowed_perc_other_blobs = allowed_perc_other_blobs
        self.mask_dilation = mask_dilation

    def __str__(self):
        return f"CL-{self.cropping_lable}_" \
               f"margins-{self.slice_margins}_OB-{self.allowed_perc_other_blobs}_MD-{self.mask_dilation}"


def crop_to_boxes_of_interset_cc(image_volume, labels_volume, params):
    crops = []

    binary_volume = labels_volume == params.cropping_lable
    binary_volume = binary_dilation(binary_volume, iterations=params.mask_dilation)

    cc = cc3d.connected_components(binary_volume)
    for label, image in cc3d.each(cc, binary=True, in_place=True):
        nwhere = np.where(image)
        relevant_ranges = tuple(slice(x.min(), x.max() + 1) for x in nwhere)
        values, counts = np.unique(cc[relevant_ranges].flatten(), return_counts=True)
        label_idx = np.where(values == label)[0][0]
        other_non_bg_occurances = np.delete(counts, [0, label_idx]).sum()

        # Check if too many overlapping blobs
        if other_non_bg_occurances / cc[relevant_ranges].size > params.allowed_perc_other_blobs:
            continue

        ranges_with_margins = tuple(slice(
                            max(0, x.min() - params.slice_margins[i]),
                            x.max() + 1 + params.slice_margins[i])
                                    for i, x in enumerate(nwhere))
        location_string = "-".join([str((x.stop + x.start) // 2) for x in ranges_with_margins])
        crops.append((image_volume[ranges_with_margins], labels_volume[ranges_with_margins], location_string))

    return crops

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


def create_dataset(data_paths, min_sizes=(4, 10, 10), normalize_axial_mm=None, crop_params=None):
    """"
    Create a dataset of 3d crops of tumors with margins
    param: crop_params: optional parameters for cropped version of the data
    param: min_sizes: minimal dimensions for a volume
    param: spatial_scale: spatial scale factor
    #### param: slice_size_mm: down/up sample in z dimension to normalize the real world size between CT slices to number of mm

    """
    processed_dir = f"{dataset_name}_(MS-{min_sizes}" \
                    + (f"_MM-{normalize_axial_mm}" if normalize_axial_mm is not None else '') \
                    + (f"_Crop-{crop_params}" if crop_params is not None else '') \
                    + ")"
    new_ct_dir = os.path.join(processed_dir, 'ct')
    new_seg_dir = os.path.join(processed_dir, 'seg')

    os.makedirs(new_ct_dir, exist_ok=True)
    os.makedirs(new_seg_dir, exist_ok=True)

    dropped_blobs = 0
    spacings = []
    for ct_filepath, gt_fname in tqdm(data_paths):
        # Read data
        ct = sitk.ReadImage(ct_filepath, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(gt_fname, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        assert (seg_array.shape == ct_array.shape)

        spacings.append(ct.GetSpacing()[-1])

        # Crop blobs and save with the same
        if crop_params is not None:
            all_blobs = crop_to_boxes_of_interset_cc(ct_array, seg_array, crop_params)
        else:
            all_blobs = [(ct_array, seg_array, "")]

        for blob_idx, (ct_array, seg_array, location_string) in enumerate(all_blobs):

            # Resample
            if normalize_axial_mm is not None:
                new_dims = (ct.GetSpacing()[-1] / normalize_axial_mm, 1, 1)
                ct_array = ndimage.zoom(ct_array, new_dims, order=3)
                seg_array = ndimage.zoom(seg_array, new_dims, order=0)

            # drop small volumes
            if np.any(ct_array.shape < np.array(min_sizes)):
                print(f"dropped_blob in case {ct_filepath} with shape {ct_array.shape}")
                dropped_blobs += 1
                continue

            # Finally save data
            fname = f"{os.path.basename(os.path.splitext(ct_filepath)[0])}"

            if location_string:
                fname += f"-({location_string}).nii"
            else:
                fname += f"-{blob_idx}.nii"

            new_ct = sitk.GetImageFromArray(ct_array)
            new_ct.SetSpacing(ct.GetSpacing())
            sitk.WriteImage(new_ct, os.path.join(new_ct_dir, fname))

            new_seg = sitk.GetImageFromArray(seg_array.astype(np.uint8))
            new_seg.SetSpacing(ct.GetSpacing())
            sitk.WriteImage(new_seg, os.path.join(new_seg_dir, fname).replace(f'volume', f'segmentation'))

    print(f"Done. Dropped blobs: {dropped_blobs}. Avg spacing: {np.mean(spacings)}")


if __name__ == '__main__':
    # dataset_name = 'KiTS2019'
    dataset_name = 'LiTS2017'
    if dataset_name == 'KiTS2019':
        data_paths = get_KiTS2019_paths('/home/ariel/projects/MedicalImageSegmentation/data/KidneyTumorSegmentation2019/train')
    elif dataset_name == 'LiTS2017':
        data_paths = get_LiTS2017_paths('/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/train')
    crop_params = cropping_parameterss(cropping_lable=1, slice_margins=(1, 1, 1), mask_dilation=11)
    create_dataset(data_paths, min_sizes=(3, 15, 15), normalize_axial_mm=2, crop_params=crop_params)
