import os
from time import time

import numpy as np
from torchvision.utils import save_image

import torch

from losses import compute_segmentation_score, TverskyScore, compute_IOU
from utils import overlay


def evaluate(model, dataloader, device, outputs_dir, n_plotted_volumes=2):
    model.eval()

    os.makedirs(outputs_dir, exist_ok=True)

    total_dice = 0
    total_iou = 0
    total_slices_per_sec = 0
    # iterate over the validation set
    for b_idx, (ct_volume, gt_volume) in enumerate(dataloader):
        assert(ct_volume.shape[0] == 1)
        ct_volume = ct_volume.to(device=device, dtype=torch.float32)
        gt_volume = gt_volume.to(device=device, dtype=torch.long)

        start = time()
        pred_volume = model.predict_volume(ct_volume)
        total_slices_per_sec += ct_volume.shape[-3] / (time() - start)

        dice_score = compute_segmentation_score(pred_volume, gt_volume.unsqueeze(1).long(), TverskyScore(0.5, 0.5))
        total_dice += dice_score

        iou_score = compute_segmentation_score(pred_volume, gt_volume.unsqueeze(1).long(), compute_IOU)
        total_iou += iou_score

        if n_plotted_volumes is None or b_idx < n_plotted_volumes:
            pred_class = pred_volume.argmax(dim=1)
            raw = overlay(ct_volume[0], gt_volume[0]*0)
            gt_vis = overlay(ct_volume[0], gt_volume[0])
            pred_vis = overlay(ct_volume[0], pred_class[0])
            imgs = torch.cat([raw, gt_vis, pred_vis], dim=-1)
            for s in range(ct_volume.shape[1]):
                save_path = os.path.join(outputs_dir, f"{b_idx}-{s}_Dice-{dice_score:.3f}_IOU-{iou_score:.3f}.png")
                save_image(imgs[s], save_path, normalize=True)

    # Fixes a potential division by zero error
    model.train()
    results = {"Dice": total_dice / len(dataloader),
               "IOU": total_iou / len(dataloader),
               "Slice/sec": total_slices_per_sec / len(dataloader)}
    return results

