import argparse
import json
import os

import numpy as np
import torch
import SimpleITK as sitk
from torchvision.transforms import InterpolationMode
from scipy import ndimage
import torch.nn.functional as F
import cc3d
from scipy.ndimage.morphology import binary_opening, binary_dilation, binary_erosion

from config import ExperimentConfigs
from datasets.ct_dataset import get_transforms, LITS2017_VALSETS
from datasets.visualize_data import write_volume_slices
from models import get_model
from torchvision.transforms import Resize
from metrics import TverskyScore, compute_Recal


def get_model_from_dir(model_dir, ckpt_name):
    config = json.load(open(f'{model_dir}/exp_configs.json'))
    config = ExperimentConfigs(**config)
    ckpt = torch.load(f'{model_dir}/{ckpt_name}.pth')

    # get model
    model = get_model(config.get_model_config())
    model.load_state_dict(ckpt['model'])
    model.to(config.device)

    return model, config


def clean_liver_prediction(pred_volume):
    """
    Keep only largest blob in binary mask
    """
    # Convert to  binary-mask
    binary_mask = pred_volume.cpu().detach().bool().numpy()

    # erode to separate loosley connected blobs
    binary_mask = binary_erosion(binary_mask, iterations=5)

    if not binary_mask.any():
        return torch.from_numpy(binary_mask)
    # Find 3D blobs
    cc = cc3d.connected_components(binary_mask)
    clean_pred_volume = max([image for (label, image) in cc3d.each(cc, binary=True, in_place=False)],
                            key=lambda x: x.sum())

    # Restore blob to original
    clean_pred_volume = binary_dilation(clean_pred_volume, iterations=5)

    clean_pred_volume = torch.from_numpy(clean_pred_volume)

    return clean_pred_volume


def read_case(ct_dir, gt_dir, fname):
    ct_nii = sitk.ReadImage(os.path.join(ct_dir, fname))
    ct_volume = sitk.GetArrayFromImage(ct_nii)  # shape (S, 512, 512)
    gt_path = os.path.join(gt_dir, fname.replace('volume', 'segmentation'))

    gt_volume = sitk.GetArrayFromImage(sitk.ReadImage(gt_path, sitk.sitkUInt8)) if os.path.exists(gt_path) else None

    return ct_volume, gt_volume, ct_nii.GetSpacing()


class TwoStepsSegmentor:
    def __init__(self, liver_segmentation_model_dir, tumor_segmentation_model_dir):
        self.liver_segmentation_model, self.liver_segmentation_cfg = get_model_from_dir(liver_segmentation_model_dir, 'best')
        _, self.liver_segmentation_transforms = get_transforms(self.liver_segmentation_cfg.get_data_config())

        # Load Tumor model and cnfigs
        self.tumor_model, self.tumor_cfg = get_model_from_dir(tumor_segmentation_model_dir, 'best')
        _, self.tumor_transforms = get_transforms(self.tumor_cfg.get_data_config())

    def predict(self, cropped_ct):
        # Predict liver
        input_ct = self.liver_segmentation_transforms((cropped_ct, np.ones_like(cropped_ct)))[0].unsqueeze(0).float().cuda()
        predicted_liver_mask = self.liver_segmentation_model.predict_volume(input_ct)[0].argmax(0).cpu()
        predicted_liver_mask = clean_liver_prediction(predicted_liver_mask).long()

        # Predict tumors
        input_ct = self.tumor_transforms((cropped_ct, np.ones_like(cropped_ct)))[0].unsqueeze(0).float().cuda()  # shape (S, 128, 128)
        print(input_ct.shape)
        predicted_tumor_mask = self.tumor_model.predict_volume(input_ct)[0].argmax(0).cpu()  # shape (S, 128, 128)

        multiclass_mask = predicted_liver_mask
        multiclass_mask[torch.logical_and(multiclass_mask == 1, predicted_tumor_mask == 1)] = 2

        return multiclass_mask


class OneStepsSegmentor:
    def __init__(self, multiclass_model_dir):
        self.multiclass_model, self.multiclass_cfg = get_model_from_dir(multiclass_model_dir, 'best')
        _, self.multiclass_transforms = get_transforms(self.multiclass_cfg.get_data_config())

    def predict(self, cropped_ct):
        input_ct = self.multiclass_transforms((cropped_ct, np.ones_like(cropped_ct)))[0].unsqueeze(0).float().cuda()  # shape (S, 128, 128)
        multiclass_mask = self.multiclass_model.predict_volume(input_ct)[0].argmax(0).cpu()  # shape (S, 128, 128)

        return multiclass_mask


def inference(ct_path, gt_path, liver_localization_model_dir, multiclass_segmentation_model_dir,  normalized_mms=None, liver_crop_padding=(3, 20,20)):
    with torch.no_grad():
        
        outputs_dir = os.path.join(os.path.dirname(ct_path), 'end2end_prediction')
        os.makedirs(outputs_dir, exist_ok=True)

        liver_localization_model, liver_localization_cfg = get_model_from_dir(liver_localization_model_dir, 'best')
        _, liver_localization_transforms = get_transforms(liver_localization_cfg.get_data_config())

        segmentor = OneStepsSegmentor(multiclass_segmentation_model_dir)

        liver_dice_scores = []
        tumor_dice_scores = []
        tumor_recalls = []
        for fname in os.listdir(ct_path):
            print(f"Case-{fname}")
            ct_volume, gt_volume, spacing = read_case(ct_path, gt_path, fname)

            # localize liver
            liver_input = liver_localization_transforms((ct_volume.copy(), np.ones_like(ct_volume).astype(np.uint8)))[0].unsqueeze(0).float().cuda()
            predicted_liver_mask = liver_localization_model.predict_volume(liver_input)[0].argmax(0).cpu()                                        # shape (S, 128, 128)

            # Clean liver and restore to origial size
            predicted_liver_mask = clean_liver_prediction(predicted_liver_mask)                                                      # shape (S, 128, 128)
            predicted_liver_mask = Resize(ct_volume.shape[-2:], interpolation=InterpolationMode.NEAREST)(predicted_liver_mask)       # shape (S, 512, 512)

            # Crop around liver for tumor segmentaiton                                                                           # shape (S, h, w)
            nwhere = np.where(predicted_liver_mask)
            liver_crop = tuple([slice(max(0, x.min() - liver_crop_padding[i]), x.max() + liver_crop_padding[i]) for i, x in enumerate(nwhere)])
            cropped_ct = ct_volume[liver_crop]                                                                                      # shape (S, h, w)

            if normalized_mms is not None:
                cropped_ct = ndimage.zoom(cropped_ct, (spacing[-1] / normalized_mms, 1, 1), order=3)

            multiclass_mask = segmentor.predict(cropped_ct)

            # Restore input resolution
            multiclass_mask = F.interpolate(multiclass_mask.float().unsqueeze(0).unsqueeze(0), size=ct_volume[liver_crop].shape[-3:], mode='nearest')[0,0].long()

            # Create final prediction mask
            final_mask = torch.zeros(ct_volume.shape).long()
            final_mask[liver_crop] = multiclass_mask

            # write segmentation map
            new_seg = sitk.GetImageFromArray(final_mask, sitk.sitkInt8)
            new_seg.SetSpacing(spacing)
            sitk.WriteImage(new_seg, os.path.join(outputs_dir, fname.replace('volume', 'segmentation')))

            # # Compute dice
            if gt_volume is not None:
                gt_volume = torch.from_numpy(gt_volume)
                liver_score = TverskyScore(0.5,0.5)((final_mask != 0).long().unsqueeze(0), (gt_volume != 0).long().unsqueeze(0), torch.ones_like(gt_volume).unsqueeze(0))
                tumor_score = TverskyScore(0.5,0.5)((final_mask == 2).long().unsqueeze(0), (gt_volume == 2).long().unsqueeze(0), torch.ones_like(gt_volume).unsqueeze(0))
                tumor_recall = compute_Recal((final_mask == 2).long().unsqueeze(0), (gt_volume == 2).long().unsqueeze(0), torch.ones_like(gt_volume).unsqueeze(0))
                liver_dice_scores.append(liver_score)
                tumor_dice_scores.append(tumor_score)
                tumor_recalls.append(tumor_recall)

                # dump debug images
                ct_volume = torch.from_numpy(np.clip(ct_volume, -100, 400).astype(float))
                write_volume_slices(ct_volume, [final_mask, gt_volume], os.path.join(outputs_dir, "e2e_test", f"{os.path.splitext(fname)[0]}_{liver_score.item():.2f}_{tumor_score.item():.2f}"))

                print(f"AVG Dice per case: Liver: {np.mean(liver_dice_scores)}, Tumor: {np.mean(tumor_dice_scores)}, Tumor-Recall: {np.mean(tumor_recalls)}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Lits2017 dataset')
    parser.add_argument('ct_dir')
    parser.add_argument('--gt_dir', default="", help='If GT is not specified no Dice score is computed')
    parser.add_argument('--localization_model_dir', default='trained_models/liver_localization/VGGUNet_Aug_Loss(0.0Dice+0.0WCE+1.0CE)_V-A')
    parser.add_argument('--segmentation_model_dir', default='trained_models/multiclass_segmentation/VGGUNet2_5D_Aug_FNE-0.5_Loss(0.0Dice+0.0WCE+1.0CE)_V-A')
    args = parser.parse_args()

    inference(args.ct_dir, args.gt_dir, args.localization_model_dir, args.segmentation_model_dir)
