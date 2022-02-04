import os
from time import time

import numpy as np
import torch

from datasets.visualize_data import write_volume_slices
from metrics import compute_segmentation_score, TverskyScore, compute_IOU


def evaluate(model, dataloader, device, volume_crieteria, outputs_dir=None):
    model.eval()
    with torch.no_grad():
        total_slices_per_sec = 0
        dice_scores = []
        iou_scores = []
        loss_values = []
        # iterate over the validation set
        for b_idx, sample in enumerate(dataloader):
            ct_volume = sample['ct'].to(device=device, dtype=torch.float32)
            gt_volume = sample['gt'].to(device=device, dtype=torch.long)
            mask_volume = sample['mask'].to(device=device, dtype=torch.bool)
            assert(ct_volume.shape[0] == 1)
            case_name = sample['case_name'][0]

            start = time()
            pred_volume = model.predict_volume(ct_volume)
            total_slices_per_sec += ct_volume.shape[-3] / (time() - start)

            dice_per_class = compute_segmentation_score(TverskyScore(0.5, 0.5), pred_volume, gt_volume.unsqueeze(1), mask_volume.unsqueeze(1), return_per_class=True)
            loss = volume_crieteria(pred_volume, gt_volume, mask_volume)
            dice_scores.append(dice_per_class)
            loss_values.append(loss.item())

            # plot volume
            if outputs_dir is not None:
                dir_path = os.path.join(outputs_dir, f"Case-{case_name}_Dice-{[f'{x:.3f}' for x in dice_per_class]}")
                write_volume_slices(ct_volume[0], [pred_volume.argmax(dim=1)[0], gt_volume[0]], dir_path)

        dice_scores = torch.stack(dice_scores).mean(0)
        report = dict()
        for i in range(1, len(dice_scores)):
            report[f'Dice-class-{i}'] = dice_scores[i]

        report["Loss"] = np.mean(loss_values)
        # report["Slice/sec"] = total_slices_per_sec / len(dataloader)
        model.train()
        return report


