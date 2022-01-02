import os
from torchvision.utils import save_image

import torch

from dice_score import compute_dice_score
from utils import overlay


def evaluate(model, dataloader, device, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)

    dice_score = 0
    # iterate over the validation set
    for b_idx, (ct_volume, gt_volume) in enumerate(dataloader):
        ct_volume = ct_volume.to(device=device, dtype=torch.float32)
        gt_volume = gt_volume.to(device=device, dtype=torch.long)
        pred_volume = model.predict_volume(ct_volume)

        dice_score += compute_dice_score(pred_volume, gt_volume.unsqueeze(1).long())

        if b_idx < 2:
            pred_class = pred_volume.argmax(dim=1)
            for i in range(ct_volume.shape[0]):
                for s in range(ct_volume.shape[1]):
                    gt_vis = overlay(ct_volume[i, s].unsqueeze(0), gt_volume[i, s].unsqueeze(0))
                    pred_vis = overlay(ct_volume[i, s].unsqueeze(0), pred_class[i, s].unsqueeze(0))
                    img = torch.cat([gt_vis, pred_vis], dim=-1)
                    save_image(img, os.path.join(outputs_dir, f"{b_idx}-{i}-{s}.png"), normalize=True)

    # Fixes a potential division by zero error
    return dice_score / len(dataloader)


def normalize_img(im):
    img = im - im.min()
    im = im / im.max()
    return im