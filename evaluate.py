import os
from time import time

import torch

from datasets.visualize_data import write_volume_slices
from metrics import compute_segmentation_score, TverskyScore, compute_IOU
from config import device


def evaluate(model, dataloader, outputs_dir=None):
    model.eval()

    total_dice = 0
    total_iou = 0   
    total_slices_per_sec = 0
    # iterate over the validation set
    results_report = dict()
    for b_idx, sample in enumerate(dataloader):
        ct_volume = sample['ct'].to(device=device, dtype=torch.float32)
        gt_volume = sample['gt'].to(device=device, dtype=torch.long)
        mask_volume = sample['mask'].to(device=device)
        assert(ct_volume.shape[0] == 1)
        case_name = sample['case_name'][0]

        start = time()
        pred_volume = model.predict_volume(ct_volume)
        total_slices_per_sec += ct_volume.shape[-3] / (time() - start)

        dice_per_class = compute_segmentation_score(TverskyScore(0.5, 0.5), pred_volume, gt_volume.unsqueeze(1), mask_volume.unsqueeze(1), return_per_class=True)
        total_dice += dice_per_class

        iou_score = compute_segmentation_score(compute_IOU, pred_volume, gt_volume.unsqueeze(1), mask_volume.unsqueeze(1), return_per_class=True)
        total_iou += iou_score

        # plot volume
        if outputs_dir:
            dir_path = os.path.join(outputs_dir, f"Case-{case_name}_Dice-{[f'{x:.3f}' for x in dice_per_class]}")
            write_volume_slices(ct_volume[0], [pred_volume.argmax(dim=1)[0], gt_volume[0]], dir_path)

    # Fixes a potential division by zero error
    model.train()
    total_dice /= len(dataloader)
    total_iou /= len(dataloader)
    results_report.update({"Slice/sec": total_slices_per_sec / len(dataloader)})
    results_report[f"Dice-non-bg"] = total_dice[1:].mean()
    for i in range(1, len(total_dice)):
        results_report[f"Dice-class-{i}"] = total_dice[i].item()
        results_report[f"IOU-class-{i}"] = total_iou[i].item()
    return results_report

