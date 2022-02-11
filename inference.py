import json
import os

import numpy as np
import torch
import SimpleITK as sitk
from torchvision.transforms import InterpolationMode

import cc3d
from scipy.ndimage.morphology import binary_opening, binary_dilation, binary_erosion

from config import ExperimentConfigs
from datasets.ct_dataset import get_transforms
from datasets.visualize_data import write_volume_slices
from models import get_model
from torchvision.transforms import Resize


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


def main():
    data_path = '/home/ariel/projects/MedicalImageSegmentation/data/LiverTumorSegmentation/test/ct'

    # Load Liver model and cnfigs
    liver_model_dir = '/mnt/storage_ssd/train_outputs/cluster_LiverTraining/VGGUNet_Aug_Loss(0.0Dice+0.0WCE+1.0CE)_V-A'
    liver_model, liver_cfg = get_model_from_dir(liver_model_dir, 'step-30000')
    _, liver_transforms = get_transforms(liver_cfg.get_data_config())

    # Load Tumor model and cnfigs
    tumor_model_dir = '/mnt/storage_ssd/train_outputs/cluster_training_batch-3-leftovers_2/VGGUNet2_5D_Aug_Elastic_MaskBg_FNE_Loss(0.0Dice+0.0WCE+1.0CE)_V-A'
    tumor_model, tumor_cfg = get_model_from_dir(tumor_model_dir, 'step-20000')
    _, tumor_transforms = get_transforms(tumor_cfg.get_data_config())

    os.makedirs(os.path.join(liver_model_dir, 'seg'), exist_ok=True)

    for fname in os.listdir(data_path):
        ct_nii = sitk.ReadImage(os.path.join(data_path, fname), sitk.sitkInt16)
        ct_volume = sitk.GetArrayFromImage(ct_nii)                                                             # shape (S, 512, 512)

        # predict liver
        liver_input = liver_transforms((ct_volume, np.ones_like(ct_volume)))[0].unsqueeze(0).float().cuda()    # shape (1, S, 128, 128)
        predicted_liver_mask = liver_model.predict_volume(liver_input)[0].argmax(0).cpu()                      # shape (S, 128, 128)

        # Clean liver and restore to origial size
        predicted_liver_mask = clean_liver_prediction(predicted_liver_mask)                                    # shape (S, 128, 128)
        predicted_liver_mask = Resize(ct_volume.shape[-2:], interpolation=InterpolationMode.NEAREST)(predicted_liver_mask) # shape (S, 512, 512)

        # Crop around liver for tumor segmentaiton
        liver_crop = [slice(max(0, x.min() - 10), x.max() + 10) for x in np.where(predicted_liver_mask)]
        tumor_input = ct_volume[liver_crop]                                                                    # shape (S, h, w)

        # Predict tumors
        tumor_input = tumor_transforms((tumor_input, np.ones_like(tumor_input)))[0].unsqueeze(0).float().cuda() # shape (S, 128, 128)
        predicted_tumor_mask = tumor_model.predict_volume(tumor_input)[0].argmax(0).cpu()                       # shape (S, 128, 128)

        # Restore input resolution
        predicted_tumor_mask = Resize(ct_volume[liver_crop].shape[-2:], interpolation=InterpolationMode.NEAREST)(predicted_tumor_mask) # shape (S, h, w)

        # Create final prediction mask
        final_mask = torch.zeros_like(predicted_liver_mask).long()
        # final_mask = predicted_liver_mask
        final_mask[liver_crop][predicted_tumor_mask == 1] = 2

        # Clean tumor noise using Liver mask
        final_mask[predicted_liver_mask == 0] = 0

        # dump debug images
        ct_volume = np.clip(ct_volume, -100, 400)
        write_volume_slices(torch.from_numpy(ct_volume), [final_mask], os.path.join(liver_model_dir, "debug_test", os.path.splitext(fname)[0]))

        # write segmentation map
        new_seg = sitk.GetImageFromArray(final_mask, sitk.sitkInt8)
        new_seg.SetSpacing(ct_nii.GetSpacing())
        sitk.WriteImage(new_seg, os.path.join(liver_model_dir, 'seg', fname.replace('test-volume', 'test-segmentation')))

if __name__ == '__main__':
    main()