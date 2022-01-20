import os
import sys

import numpy as np
import torch

import scipy.ndimage.morphology
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from datasets.data_utils import read_volume
from datasets.visualize_data import write_volume_slices
from metrics import TverskyScore


def read_case(data_root, case, axis_slice=None):
    """
    Read case and slice it
    """
    ct = read_volume(os.path.join(data_root, 'ct', f"volume-{case}.nii"))
    gt = read_volume(os.path.join(data_root, 'seg', f"segmentation-{case}.nii")).astype(np.uint8)
    if type(axis_slice) == int:
        ct = ct[axis_slice:axis_slice+1]
        gt = gt[axis_slice:axis_slice+1]
    elif type(axis_slice) == tuple:
        ct = ct[axis_slice[0]:axis_slice[1]]
        gt = gt[axis_slice[0]:axis_slice[1]]
    return ct, gt


def focus_on_liver(ct, gt):
    """
    Crop 3d around liver
    """
    slicing = tuple(slice(x.min() - 10 , x.max() +11) for x in np.where(gt != 0))
    return ct[slicing], gt[slicing]


def normalize(ct, target_min_v=0, target_max_v=255):
    """
    Strech intencities into [target_min_v, target_max_v]
    """
    image_min = ct.min()
    image_max = ct.max()
    result = (ct - image_min) / (image_max - image_min) * (target_max_v - target_min_v) + target_min_v
    return result.astype(np.uint8)


def hist_equalization(image, mask):
    image_histogram, bins = np.histogram(image[mask], bins=256, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image[mask] = np.interp(image[mask], bins[:-1], cdf)
    return image


def normalize_intencities(ct, mask):
    if np.any(mask):
        ct = ct.clip(np.percentile(ct[mask], 5), np.percentile(ct[mask], 95))
    ct = normalize(ct)
    return ct


def get_dice(pred, gt):
    pred_t = torch.from_numpy(pred).unsqueeze(0)
    gt_t = torch.from_numpy(gt)
    target = (gt_t == 2).float().unsqueeze(0)
    whole_liver_mask = gt != 0
    return TverskyScore(0.5, 0.5)(pred_t, target, mask=whole_liver_mask).item()


def plot_hists(ct, masks, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for (mask, label) in masks:
        if np.any(mask):
            nbins = np.unique(ct[mask]).size // 3
            plt.hist(ct[mask], bins=nbins, alpha=0.5, density=True, label=label)
    plt.legend()
    plt.savefig(path)
    plt.clf()


def predict_slice(ct, liver_mask, t):
    """
    Run thresholding algorithm on normalized intencities
    """
    if not liver_mask.any():
        return np.zeros_like(liver_mask)

    ct[~liver_mask] = 255
    ct = gaussian_filter(ct, sigma=3, truncate=3)

    pred = np.where(ct > t, 0, 1)

    pred = scipy.ndimage.morphology.binary_erosion(pred, iterations=1)
    pred = scipy.ndimage.morphology.binary_dilation(pred, iterations=1)
    pred = pred.astype(np.uint8)
    return pred


def predict_volume_by_slices(ct, liver_mask, t):
    pred = np.stack([predict_slice(ct[i], liver_mask[i], t=t) for i in range(len(ct))], axis=0)
    return pred


def run_on_validation_set(data_root, test_cases, t, out_dir=None):
    """Run on multiple cases and return dice-per-case"""
    dice_scores = []
    for case in test_cases:
        ct, gt = read_case(data_root, case)
        liver_mask = gt != 0
        ct = normalize_intencities(ct, liver_mask)

        pred = predict_volume_by_slices(ct.copy(), liver_mask, t=t)

        dice_score = get_dice(pred, gt)
        dice_scores.append(dice_score)
        print(f"Case {case}, dice: {dice_score}")

        # Write debug images
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plot_hists(ct, [(gt == 1, "liver"), (gt == 2, "tumor")], f"{out_dir}/Hist-case-{case}")
            write_volume_slices(torch.from_numpy(ct),  [torch.from_numpy(gt), torch.from_numpy(pred)], f"{out_dir}/Slices-case-{case}")

    print(f"AVg, dice: {np.mean(dice_scores)}")


if __name__ == '__main__':
    data_path = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/raw_data'

    # Run full test
    test_cases = [19, 76, 50, 92, 88, 122, 100, 71, 23, 28, 9, 119, 39]
    run_on_validation_set(data_path, test_cases, t=120, out_dir="debug_test")