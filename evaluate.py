import os
from time import time

import numpy as np
from torchvision.utils import save_image

import torch

from losses import compute_segmentation_score, TverskyScore
from utils import overlay


def evaluate(model, dataloader, device, outputs_dir, n_plotted_volumes=2):
    model.eval()

    os.makedirs(outputs_dir, exist_ok=True)

    total_score = 0
    # iterate over the validation set
    for b_idx, (ct_volume, gt_volume) in enumerate(dataloader):
        ct_volume = ct_volume.to(device=device, dtype=torch.float32)
        gt_volume = gt_volume.to(device=device, dtype=torch.long)
        pred_volume = model.predict_volume(ct_volume)

        score = compute_segmentation_score(pred_volume, gt_volume.unsqueeze(1).long(), TverskyScore(0.5, 0.5))
        total_score += score

        if b_idx < n_plotted_volumes:
            pred_class = pred_volume.argmax(dim=1)
            for i in range(ct_volume.shape[0]):
                for s in range(ct_volume.shape[1]):
                    gt_vis = overlay(ct_volume[i, s]    .unsqueeze(0), gt_volume[i, s].unsqueeze(0))
                    pred_vis = overlay(ct_volume[i, s].unsqueeze(0), pred_class[i, s].unsqueeze(0))
                    img = torch.cat([gt_vis, pred_vis], dim=-1)
                    save_image(img, os.path.join(outputs_dir, f"{b_idx}-{i}-{s}_Score-{score:.3f}.png"), normalize=True)

    # Fixes a potential division by zero error
    model.train()
    return total_score / len(dataloader)


def test(model, dataloader, device, outputs_dir):
    model.eval()

    os.makedirs(outputs_dir, exist_ok=True)

    volume_scores = []
    pred_times = []
    # iterate over the validation set
    for b_idx, (ct_volume, gt_volume) in enumerate(dataloader):
        start = time()
        pred_volume = model.predict_volume(ct_volume.to(device).float()).cpu()
        pred_times.append(ct_volume.shape[-3] / (time() - start))

        volume_score = compute_segmentation_score(pred_volume, gt_volume.unsqueeze(1).long(), TverskyScore(0.5, 0.5))
        volume_loss = compute_segmentation_loss(pred_volume, gt_volume.unsqueeze(1).long(), TverskyScore(0.5, 0.5))
        volume_dir = f"{outputs_dir}/{b_idx}-Score-{volume_score:.3f}-Loss{volume_loss:.3f}"
        os.makedirs(volume_dir, exist_ok=True)
        volume_scores.append(volume_score)
        pred_labelmap = pred_volume.argmax(dim=1)
        for i in range(ct_volume.shape[0]):
            raw = overlay(ct_volume[i], gt_volume[i]*0)
            gt_vis = overlay(ct_volume[i], gt_volume[i])
            pred_vis = overlay(ct_volume[i], pred_labelmap[i])
            img = torch.cat([raw, gt_vis, pred_vis], dim=-1)

            for s in range(ct_volume.shape[-3]):
                pred_slice = pred_volume[...,s, :, :].unsqueeze(-3)
                gt_slice = gt_volume[...,s, :, :].unsqueeze(-3).unsqueeze(1).long()
                slice_score = compute_segmentation_score(pred_slice, gt_slice, TverskyScore(0.5, 0.5))
                slice_loss = compute_segmentation_loss(pred_slice, gt_slice, TverskyScore(0.5, 0.5))
                save_image(img[s], f"{volume_dir}/{i}-{s}_Score{slice_score:.3f}_Loss{slice_loss:.3f}.png", normalize=True)

    model.train()

    return np.mean(volume_scores), np.mean(pred_times)
    # Fixes a potential division by zero error
