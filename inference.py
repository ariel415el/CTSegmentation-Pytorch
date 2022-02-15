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
from metrics import TverskyScore


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

    # Find 3D blobs
    cc = cc3d.connected_components(binary_mask)
    clean_pred_volume = max([image for (label, image) in cc3d.each(cc, binary=True, in_place=False)],
                            key=lambda x: x.sum())

    # Restore blob to original
    clean_pred_volume = binary_dilation(clean_pred_volume, iterations=5)

    clean_pred_volume = torch.from_numpy(clean_pred_volume)

    return clean_pred_volume

def read_case(ct_dir, gt_dir, fname):
    ct_nii = sitk.ReadImage(os.path.join(ct_dir, fname), sitk.sitkInt16)
    ct_volume = sitk.GetArrayFromImage(ct_nii)  # shape (S, 512, 512)
    gt_volume = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_dir, fname.replace('volume', 'segmentation'))))

    return ct_volume, gt_volume, ct_nii.GetSpacing()


def main():
    with torch.no_grad():
        normalized_mms = 2
        padding = 10
        ct_path = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/train/ct'
        gt_path = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/train/seg'
        liver_localization_model_dir = '/mnt/storage_ssd/train_outputs/liver_batch1/VGGUNet_Aug_Loss(0.0Dice+0.0WCE+1.0CE)_V-A'
        liver_segmentation_model_dir = '/mnt/storage_ssd/train_outputs/liver_batch_cropped/VGGUNet_Aug_FNE-0.5_Loss(1.0Dice+0.0WCE+1.0CE)_V-A/'
        tumor_model_dir = '/mnt/storage_ssd/train_outputs/Best_models/UNet3D_Aug_Elastic_MaskBg_FNE-0.5_Loss(1.0Dice+0.0WCE+1.0CE)_V-A'
        os.makedirs(os.path.join(tumor_model_dir, 'seg'), exist_ok=True)

        # Load Liver model and cnfigs
        liver_localization_model, liver_localization_cfg = get_model_from_dir(liver_localization_model_dir, 'step-30000')
        _, liver_localization_transforms = get_transforms(liver_localization_cfg.get_data_config())

        # Load Liver model and cnfigs
        liver_segmentation_model, liver_segmentation_cfg = get_model_from_dir(liver_segmentation_model_dir, 'best')
        _, liver_segmentation_transforms = get_transforms(liver_segmentation_cfg.get_data_config())

        # Load Tumor model and cnfigs
        tumor_model, tumor_cfg = get_model_from_dir(tumor_model_dir, 'best')
        _, tumor_transforms = get_transforms(tumor_cfg.get_data_config())

        liver_scores = []
        tumor_scores = []
        for i in LITS2017_VALSETS['A']:
            fname = f'volume-{i}.nii'
            ct_volume, gt_volume, spacing = read_case(ct_path, gt_path, fname)
            gt_volume = torch.from_numpy(gt_volume)

            # localize liver
            liver_input = liver_localization_transforms((ct_volume, np.ones_like(ct_volume)))[0].unsqueeze(0).float().cuda()                      # shape (1, S, 128, 128)
            predicted_liver_mask = liver_localization_model.predict_volume(liver_input)[0].argmax(0).cpu()                                        # shape (S, 128, 128)

            # Clean liver and restore to origial size
            predicted_liver_mask = clean_liver_prediction(predicted_liver_mask)                                                      # shape (S, 128, 128)
            predicted_liver_mask = Resize(ct_volume.shape[-2:], interpolation=InterpolationMode.NEAREST)(predicted_liver_mask)       # shape (S, 512, 512)
            # predicted_liver_mask = gt_volume.bool()

            # Crop around liver for tumor segmentaiton
            liver_crop = [slice(max(0, x.min() - padding), x.max() + padding) for x in np.where(predicted_liver_mask)]
            cropped_ct = ct_volume[liver_crop]                                                                                      # shape (S, h, w)

            cropped_ct = ndimage.zoom(cropped_ct, (spacing[-1] / normalized_mms, 1, 1), order=3)

            # Predict liver
            input_ct = liver_segmentation_transforms((cropped_ct, np.ones_like(cropped_ct)))[0].unsqueeze(0).float().cuda()
            predicted_liver_mask = liver_segmentation_model.predict_volume(input_ct)[0].argmax(0).cpu()
            predicted_liver_mask = clean_liver_prediction(predicted_liver_mask).long()

            # Predict tumors
            input_ct = tumor_transforms((cropped_ct, np.ones_like(cropped_ct)))[0].unsqueeze(0).float().cuda()                  # shape (S, 128, 128)
            predicted_tumor_mask = tumor_model.predict_volume(input_ct)[0].argmax(0).cpu()                                      # shape (S, 128, 128)

            multiclass_mask = predicted_liver_mask
            multiclass_mask[torch.logical_and(multiclass_mask == 1, predicted_tumor_mask == 1)] = 2

            # Restore input resolution
            multiclass_mask = F.interpolate(multiclass_mask.float().unsqueeze(0).unsqueeze(0), size=ct_volume[liver_crop].shape[-3:], mode='nearest')[0,0].long()
            # multiclass_mask = Resize(cropped_ct.shape[-3:], interpolation=InterpolationMode.NEAREST)(multiclass_mask) # shape (S, h, w)

            # Create final prediction mask
            final_mask = torch.zeros(ct_volume.shape).long()
            final_mask[liver_crop] = multiclass_mask

            # write segmentation map
            new_seg = sitk.GetImageFromArray(final_mask, sitk.sitkInt8)
            new_seg.SetSpacing(spacing)
            sitk.WriteImage(new_seg, os.path.join(tumor_model_dir, 'seg', f'test-segmentation-{i}.nii'))

            # # Compute dice
            liver_score = TverskyScore(0.5,0.5)((final_mask != 0).long().unsqueeze(0), (gt_volume != 0).long().unsqueeze(0), torch.ones_like(gt_volume).unsqueeze(0))
            tumor_score = TverskyScore(0.5,0.5)((final_mask == 2).long().unsqueeze(0), (gt_volume == 2).long().unsqueeze(0), torch.ones_like(gt_volume).unsqueeze(0))
            liver_scores.append(liver_score)
            tumor_scores.append(tumor_score)

            print(f"Dice per case: Liver: {np.mean(liver_scores)}, Tumor: {np.mean(tumor_scores)}")

            # dump debug images
            ct_volume = torch.from_numpy(np.clip(ct_volume, -100, 400))
            write_volume_slices(ct_volume[liver_crop], [final_mask[liver_crop], gt_volume[liver_crop]], os.path.join(tumor_model_dir, "debug_test", f"{os.path.splitext(fname)[0]}_{liver_score.item():.2f}_{tumor_score.item():.2f}"))

        print(f"AVG Dice per case: Liver: {np.mean(liver_scores)}, Tumor: {np.mean(tumor_scores)}")

def debug():
    from datasets.data_utils import read_volume
    ct_nii = sitk.ReadImage('/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/train/ct/volume-19.nii', sitk.sitkInt16)
    full_case_ct = sitk.GetArrayFromImage(ct_nii)
    full_case_gt = read_volume('/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/train/seg/segmentation-19.nii')

    new_dims = (ct_nii.GetSpacing()[-1] / 2, 1, 1)
    from scipy import ndimage
    full_case_ct = ndimage.zoom(full_case_ct, new_dims, order=3)
    full_case_gt = torch.from_numpy(ndimage.zoom(full_case_gt, new_dims, order=0))

    liver_crop = [slice(max(0, x.min() - 10), x.max() + 10) for x in np.where(full_case_gt != 0)]
    full_case_ct = full_case_ct[liver_crop]
    full_case_gt = full_case_gt[liver_crop]

    cropped_case_ct = read_volume('/home/ariel/projects/MedicalImageSegmentation/CTSegmentation-Pytorch/datasets/LiTS2017_LiverCrop_2mm/ct/volume-19-(418-275-196).nii')
    cropped_case_gt = read_volume('/home/ariel/projects/MedicalImageSegmentation/CTSegmentation-Pytorch/datasets/LiTS2017_LiverCrop_2mm/seg/segmentation-19-(418-275-196).nii')

    print(full_case_ct.shape, full_case_gt.shape)
    print(cropped_case_ct.shape, cropped_case_gt.shape)



if __name__ == '__main__':
    # debug()
    main()