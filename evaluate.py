import os
from torchvision.utils import save_image

import torch

from dice_score import compute_dice_score
from utils import overlay


def evaluate(net, dataloader, device, outputs_dir):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    os.makedirs(outputs_dir, exist_ok=True)

    # iterate over the validation set
    for b_idx, (ct_volumes, gt_volumes) in enumerate(dataloader):
        pred_volume = []
        for i in range(ct_volumes.shape[1]):
            image = ct_volumes[:, i].to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred_volume.append(net(image.unsqueeze(1)))

        pred_volume = torch.stack(pred_volume, dim=2).cpu()
        dice_score = compute_dice_score(pred_volume, gt_volumes.unsqueeze(1).long())

        if b_idx < 5:
            pred_class = pred_volume.argmax(dim=1)
            for i in range(ct_volumes.shape[0]):
                for s in range(ct_volumes.shape[1]):
                    gt_vis = overlay(ct_volumes[i, s].unsqueeze(0), gt_volumes[i, s].unsqueeze(0))
                    pred_vis = overlay(ct_volumes[i, s].unsqueeze(0), pred_class[i, s].unsqueeze(0))
                    img = torch.cat([gt_vis, pred_vis], dim=-1)
                    save_image(img, os.path.join(outputs_dir, f"{b_idx}-{i}-{s}.png"), normalize=True)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


def normalize_img(im):
    img = im - im.min()
    im = im / im.max()
    return im