import torch
import torch.nn.functional as F


def dice_score(mat1, mat2):
    """
    :param mat1: binary matrix of shape (b, H, W)
    :param mat2: binary matrix of shape (b, H, W)
    :return:
    """
    epsilon = 1e-6
    tp = (mat1 * mat2).sum()
    sets_sum = mat1.sum() + mat2.sum()
    if sets_sum.item() == 0:
       sets_sum = tp
    dice = (2 * tp + epsilon) / (sets_sum + epsilon)
    return dice


def per_class_dice(pred_volume, segmentation_volume, drop_bg_class=False):
    """
    :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
    :param segm_map: uint8 array of shape (1, 1, slices, H, W) containing segmentation labels
    """
    assert(pred_volume.shape[0] == 1)

    n_class = pred_volume.shape[1]
    gt_1hot_volume = F.one_hot(segmentation_volume[:, 0], n_class).permute(0, 4, 1, 2, 3).float()

    per_channel_dice = []
    for c in range(n_class):
        per_channel_dice.append(dice_score(pred_volume[0, c], gt_1hot_volume[0, c]))

    if drop_bg_class:
        per_channel_dice = per_channel_dice[1:]

    return torch.stack(per_channel_dice).mean()


def compute_dice_loss(pred_volume, segmentation_volume):
    """
    :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
    :param segm_map: uint8 array of shape (1, 1, slices, H, W) containing segmentation labels
    """
    pred_volume = F.softmax(pred_volume, dim=1).float()
    dice = per_class_dice(pred_volume, segmentation_volume)
    return 1 - dice


def compute_dice_score(pred_volume, segmentation_volume):
    """
    :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
    :param segm_map: uint8 array of shape (1, 1, slices, H, W) containing segmentation labels
    """
    pred_map_volume = torch.argmax(pred_volume, dim=1, keepdim=True)
    pred_volume = F.one_hot(pred_map_volume[:, 0], pred_volume.shape[1]).permute(0, 4, 1, 2, 3)
    return per_class_dice(pred_volume, segmentation_volume, drop_bg_class=True)
