import os

import numpy as np
import torch

import scipy.ndimage.morphology
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from datasets.data_utils import read_volume
from datasets.visualize_data import overlay, write_volume_slices
from metrics import TverskyScore


def read_case(data_root, case, axis_slice=None):
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
    slicing = tuple(slice(x.min() - 10 , x.max() +11) for x in np.where(gt != 0))
    return ct[slicing], gt[slicing]


def normalize(ct, target_min_v=0, target_max_v=255):
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
    # nbins = np.unique(ct[masks[0][0]]).size // 3
    for (mask, label) in masks:
        if np.any(mask):
            nbins = np.unique(ct[mask]).size // 3
            plt.hist(ct[mask], bins=nbins, alpha=0.5, density=True, label=label)
    plt.legend()
    plt.savefig(path)
    plt.clf()


def predict_slice(ct, liver_mask, t):
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


def produce_hists():
    all_liver_intencities_raw = []
    all_tumor_intencities_raw = []

    all_liver_intencities_normed = []
    all_tumor_intencities_normed = []

    for case in range(8,9):
        ct = read_volume(os.path.join(data_path, 'ct', f"volume-{case}.nii"))
        gt = read_volume(os.path.join(data_path, 'seg', f"segmentation-{case}.nii")).astype(np.uint8)

        all_liver_intencities_raw.append(ct[gt == 1].copy())
        all_tumor_intencities_raw.append(ct[gt == 2].copy())

        ct = ct.clip(np.percentile(ct[gt != 0], 5), np.percentile(ct[gt != 0], 95))
        ct = normalize(ct)

        all_liver_intencities_normed.append(ct[gt == 1].copy())
        all_tumor_intencities_normed.append(ct[gt == 2].copy())

    all_liver_intencities_raw = np.concatenate(all_liver_intencities_raw)
    all_tumor_intencities_raw = np.concatenate(all_tumor_intencities_raw)
    nbins = np.unique(all_liver_intencities_raw).size // 2
    plt.hist(all_liver_intencities_raw, bins=nbins, alpha=0.5, density=True, label='raw_liver')
    plt.hist(all_tumor_intencities_raw, bins=nbins, alpha=0.5, density=True, label='raw_tumor')
    plt.legend()
    plt.show()

    all_liver_intencities_normed = np.concatenate(all_liver_intencities_normed)
    all_tumor_intencities_normed = np.concatenate(all_tumor_intencities_normed)
    nbins = np.unique(all_liver_intencities_normed).size // 2
    plt.hist(all_liver_intencities_normed, bins=nbins, alpha=0.5, density=True, label='normed_liver')
    plt.hist(all_tumor_intencities_normed, bins=nbins, alpha=0.5, density=True, label='normed_tumor')
    plt.legend()
    plt.show()


def run_on_validation_set(data_path, test_cases, t, out_dir=None):
    dice_scores = []
    for case in test_cases:
        ct, gt = read_case(data_path, case)
        liver_mask = gt != 0
        ct = normalize_intencities(ct, liver_mask)

        pred = predict_volume_by_slices(ct.copy(), liver_mask, t=t)

        dice_score = get_dice(pred, gt)
        dice_scores.append(dice_score)
        print(dice_score)

        # Write debug images
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plot_hists(ct, [(gt == 1, "liver"), (gt == 2, "tumor")], f"{out_dir}/Hist-case-{case}")
            write_volume_slices(torch.from_numpy(ct),  [torch.from_numpy(gt), torch.from_numpy(pred)], f"{out_dir}/Slices-case-{case}")

    print(np.mean(dice_scores))


def analyze_prediction_process(data_path, case, axis_slice, t, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ct_volume, gt_volume = read_case(data_path, case, axis_slice)
    ct_volume, gt_volume = focus_on_liver(ct_volume, gt_volume)
    overlayed = overlay(torch.from_numpy(ct_volume), torch.from_numpy(gt_volume)).numpy().transpose(0,2,3,1).astype(np.uint8)
    for axis_slice in range(ct_volume.shape[0]):
        ct = ct_volume[axis_slice]
        gt = gt_volume[axis_slice]
        liver_mask = gt != 0
        ct_normed = normalize_intencities(ct.copy(), liver_mask)

        ct_liver_only = np.where(~liver_mask, 255, ct_normed)
        ct_blurr = gaussian_filter(ct_liver_only, sigma=3, truncate=3)

        binary = np.where(ct_blurr > t, 0, 1)

        erosion = scipy.ndimage.morphology.binary_erosion(binary, iterations=5)
        dilation = scipy.ndimage.morphology.binary_dilation(erosion, iterations=5)
        pred = dilation.astype(np.uint8)
        dice = get_dice(pred[None, :], gt[None, :])

        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10,10))
        fig.suptitle(f'T={t}, Dice: {dice:.2f}')
        ax[0, 0].imshow(overlayed[axis_slice])
        ax[0, 0].set_title("raw")
        ax[0, 0].set_axis_off()

        ax[0, 1].imshow(ct_normed, cmap='gray')
        ax[0, 1].set_title("normalized")
        ax[0, 1].set_axis_off()

        ax[0, 2].imshow(ct_liver_only, cmap='gray')
        ax[0, 2].set_title("Liver_only")
        ax[0, 2].set_axis_off()

        ax[0, 3].imshow(ct_blurr, cmap='gray')
        ax[0, 3].set_title("Blur")
        ax[0, 3].set_title("Blur")
        ax[0, 3].set_axis_off()

        ax[1, 0].imshow(binary, cmap='gray', vmax=1)
        ax[1, 0].set_title("threshold")
        ax[1, 0].set_axis_off()

        ax[1, 1].imshow(erosion.astype(np.uint8), cmap='gray', vmax=1)
        ax[1, 1].set_title("erosion")
        ax[1, 1].set_axis_off()

        ax[1, 2].imshow(dilation, cmap='gray', vmax=1)
        ax[1, 2].set_title("dilation")
        ax[1, 2].set_axis_off()

        ax[1, 3].set_title("hists")
        ax[1, 3].hist(ct_blurr[gt == 1], bins=256, alpha=0.5, density=True, label='liver')
        ax[1, 3].hist(ct_blurr[gt == 2], bins=256, alpha=0.5, density=True, label='tumor')
        ax[1, 3].legend()

        plt.tight_layout()
        fig.savefig(f"{output_dir}/Prediction-case-{case}-{axis_slice}.png")

        plt.clf()


if __name__ == '__main__':
    data_path = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/raw_data'
    # Analyze specific slices
    for case, axis_slice in [(76, 133), (109, 449), (129, 139)]:
        analyze_prediction_process(data_path, case=case, axis_slice=axis_slice, t=120, output_dir="debug_slices")

    # Analyze entire cases
    for case in [19, 50, 92, 23, 28]:
        analyze_prediction_process(data_path, case=case, axis_slice=None, t=90, output_dir="debug")

    # Run full test
    test_cases = [19, 76, 50, 92, 88, 122, 100, 71, 23, 28, 9, 119, 39]
    run_on_validation_set(data_path, test_cases, t=120)