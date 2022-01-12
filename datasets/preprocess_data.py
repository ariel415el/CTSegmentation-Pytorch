import os
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation
import cc3d

@dataclass
class cropping_params:
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


def create_dataset(root_dir, spatial_scale=1, min_sizes=(4, 10, 10), crop_params=None, remove_liver_label=False):
    """"
    Create a dataset of 3d crops of tumors with margins
    param: remove_liver_label: ignore liver labels
    param: crop_params: optional parameters for cropped version of the data
    param: min_sizes: minimal dimensions for a volume
    param: spatial_scale: spatial scale factor
    #### param: slice_size_mm: down/up sample in z dimension to normalize the real world size between CT slices to number of mm

    """
    processed_dir = f"LiverData_(S-{spatial_scale}_MS-{min_sizes}" \
                    + (f"_RL-{remove_liver_label}" if remove_liver_label is not None else '') \
                    + (f"_CP-{crop_params}" if crop_params is not None else '') \
                    + ")"
    new_ct_dir = os.path.join(processed_dir, 'ct')
    new_seg_dir = os.path.join(processed_dir, 'seg')

    os.makedirs(new_ct_dir, exist_ok=True)
    os.makedirs(new_seg_dir, exist_ok=True)

    dropped_blobs = 0
    spacings = []
    for ct_filename in tqdm(os.listdir(os.path.join(root_dir, 'ct'))):
        # Read data
        ct = sitk.ReadImage(os.path.join(root_dir, 'ct', ct_filename), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(os.path.join(root_dir, 'seg', ct_filename.replace('volume', 'segmentation')),
                             sitk.sitkInt16)
        seg_array = sitk.GetArrayFromImage(seg)
        assert (seg_array.shape == ct_array.shape)

        spacings.append(ct.GetSpacing()[-1])
        # Manipulate values
        # ct_array = np.clip(ct_array, IMG_MIN_VAL, IMG_MAX_VAL)

        # Crop blobs and save with the same
        if crop_params is not None:
            all_blobs = crop_to_boxes_of_interset_cc(ct_array, seg_array, crop_params)
        else:
            all_blobs = [(ct_array, seg_array, "")]

        for blob_idx, (ct_array, seg_array, location_string) in enumerate(all_blobs):
            # seg_array[seg_array > 0] = 1

            if remove_liver_label:
                seg_array[seg_array == 1] = 0
                seg_array[seg_array == 2] = 1

            # Resample
            if spatial_scale != 1:
                # new_dims = (ct.GetSpacing()[-1] / slice_size_mm, spatial_scale, spatial_scale)
                new_dims = (1, spatial_scale, spatial_scale)
                ct_array = ndimage.zoom(ct_array, new_dims, order=3)
                seg_array = ndimage.zoom(seg_array, new_dims, order=0)

            # drop small volumes
            if np.any(ct_array.shape < np.array(min_sizes)):
                print(f"dropped_blob in case {ct_filename} with shape {ct_array.shape}")
                dropped_blobs += 1
                continue

            # Finally save data
            fname = f"{os.path.splitext(ct_filename)[0]}"

            if location_string:
                fname += f"-({location_string}).nii"
            else:
                fname += f"-{blob_idx}.nii"
            # np.save(os.path.join(new_ct_dir, fname), ct_array)
            # np.save(os.path.join(new_seg_dir, fname).replace(f'volume', f'segmentation'), seg_array)

            new_ct = sitk.GetImageFromArray(ct_array)
            new_ct.SetSpacing(ct.GetSpacing())
            sitk.WriteImage(new_ct, os.path.join(new_ct_dir, fname))

            new_seg = sitk.GetImageFromArray(seg_array)
            new_seg.SetSpacing(ct.GetSpacing())
            sitk.WriteImage(new_seg, os.path.join(new_seg_dir, fname).replace(f'volume', f'segmentation'))

    print(f"Done. Dropped blobs: {dropped_blobs}. Avg spacing: {np.mean(spacings)}")

if __name__ == '__main__':
    raw_data = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/raw_data'
    # # save data as is
    # create_dataset(raw_data, remove_liver_label=False, spatial_scale=0.25)

    # crop around liver and show only tumor labels
    crop_params = cropping_params(cropping_lable=1, slice_margins=(1, 1, 1), mask_dilation=11)
    create_dataset(raw_data, remove_liver_label=True, min_sizes=(3, 5, 5), crop_params=crop_params)

    # # Crop around tumors
    # crop_params = cropping_params(cropping_lable=2, slice_margins=(1, 20, 20))
    # create_dataset(raw_data, remove_liver_label=True, min_sizes=(3, 30, 30), crop_params=crop_params, )
