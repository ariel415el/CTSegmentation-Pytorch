import os

import numpy as np
import scipy.ndimage
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from classic_methods.thresholding import normalize, read_case, focus_on_liver, normalize_intencities, get_dice
from datasets.data_utils import read_volume
from datasets.visualize_data import overlay


def produce_hists(data_root, cases):
    """Plot hists of tumor vs liver intencities of multiple cases"""
    all_liver_intencities_raw = []
    all_tumor_intencities_raw = []

    all_liver_intencities_normed = []
    all_tumor_intencities_normed = []

    for case in cases:
        ct = read_volume(os.path.join(data_root, 'ct', f"volume-{case}.nii"))
        gt = read_volume(os.path.join(data_root, 'seg', f"segmentation-{case}.nii")).astype(np.uint8)

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
    plt.savefig("Raw_hists.png")
    plt.clf()

    all_liver_intencities_normed = np.concatenate(all_liver_intencities_normed)
    all_tumor_intencities_normed = np.concatenate(all_tumor_intencities_normed)
    nbins = np.unique(all_liver_intencities_normed).size // 2
    plt.hist(all_liver_intencities_normed, bins=nbins, alpha=0.5, density=True, label='normed_liver')
    plt.hist(all_tumor_intencities_normed, bins=nbins, alpha=0.5, density=True, label='normed_tumor')
    plt.legend()
    plt.savefig("Normed_hists.png")
    plt.clf()


def analyze_prediction_process(data_path, case, axis_slice, t, output_dir):
    """load sliced case and create a plit that visualizes the prediction process"""
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

    produce_hists(data_path, [19, 50, 92, 23, 28])

    # Analyze specific slices
    for case, axis_slice in [(76, 133), (109, 449), (129, 139)]:
        analyze_prediction_process(data_path, case=case, axis_slice=axis_slice, t=120, output_dir="debug_slices")

    # Analyze entire cases
    for case in [19, 50, 92, 23, 28]:
        analyze_prediction_process(data_path, case=case, axis_slice=None, t=90, output_dir="debug_volumes")