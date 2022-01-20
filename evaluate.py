import os
from time import time

from torchvision.utils import save_image

import torch

from datasets.visualize_data import write_volume_slices
from metrics import compute_segmentation_score, TverskyScore, compute_IOU


def evaluate(model, dataloader, device, outputs_dir, n_plotted_volumes=0):
    model.eval()

    total_dice = 0
    total_iou = 0   
    total_slices_per_sec = 0
    # iterate over the validation set
    for b_idx, sample in enumerate(dataloader):
        ct_volume = sample['ct'].to(device=device, dtype=torch.float32)
        gt_volume = sample['gt'].to(device=device, dtype=torch.long)
        mask_volume = sample['mask'].to(device=device)
        assert(ct_volume.shape[0] == 1)
        case_name = sample['case_name'][0]

        start = time()
        pred_volume = model.predict_volume(ct_volume)
        total_slices_per_sec += ct_volume.shape[-3] / (time() - start)

        dice_score = compute_segmentation_score(TverskyScore(0.5, 0.5), pred_volume, gt_volume.unsqueeze(1), mask_volume.unsqueeze(1), return_per_class=True)
        total_dice += dice_score

        iou_score = compute_segmentation_score(compute_IOU, pred_volume, gt_volume.unsqueeze(1), mask_volume.unsqueeze(1), return_per_class=True)
        total_iou += iou_score

        if n_plotted_volumes is None or b_idx < n_plotted_volumes:
            os.makedirs(outputs_dir, exist_ok=True)
            dir_path = os.path.join(outputs_dir, f"Case-{case_name}_Dice-{dice_score.mean():.3f}_IOU-{iou_score.mean():.3f}")
            write_volume_slices(ct_volume[0], [pred_volume.argmax(dim=1)[0], gt_volume[0]], dir_path)

    # Fixes a potential division by zero error
    model.train()
    results = {"Slice/sec": total_slices_per_sec / len(dataloader)}
    total_dice /= len(dataloader)
    total_iou /= len(dataloader)
    results[f"Dice-non-bg"] = total_dice[1:].mean()
    for i in range(1, len(total_dice)):
        results[f"Dice-class-{i}"] = total_dice[i].item()
        results[f"IOU-class-{i}"] = total_iou[i].item()
    return results

