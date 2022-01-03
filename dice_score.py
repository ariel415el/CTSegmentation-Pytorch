import torch
import torch.nn.functional as F


def dice_score(mat1, mat2):
    """
    :param mat1: binary matrix of shape (b, H, W)
    :param mat2: binary matrix of shape (b, H, W)
    :return:
    """
    tp = (mat1 * mat2).sum()
    T_plus_P = mat1.sum() + mat2.sum()
    dice = (2 * tp) / (T_plus_P + 1e-6)
    dice = torch.clip(dice, 0, 1)
    return dice


def tversky_score(pred_mat, gt_map, fp_weight=0.5, fn_weight=0.5):
    """
    Prioritize FN weight in segmentation map prediction.
    if the weights are booth 0.5 you get exactly the dice score (f1)
    :param mat1: binary matrix of shape (b, H, W)
    :param mat2: binary matrix of shape (b, H, W)
    :return:
    """
    tp = (pred_mat * gt_map).sum()
    fp = (pred_mat * (1 - gt_map)).sum()
    fn = ((1 - pred_mat) * (gt_map)).sum()
    if gt_map.sum() == 0:
        denominator = tp
    else:
        denominator = tp + fp * fp_weight + fn * fn_weight
    score = (tp + 1e-6) / (denominator + 1e-6)
    dice = torch.clip(score, 0, 1)
    return dice


def per_class_score(pred_volume, segmentation_volume, loss_func, drop_bg_class=False):
    """
    :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
    :param segm_map: uint8 array of shape (1, 1, slices, H, W) containing segmentation labels
    """
    assert (pred_volume.shape[0] == 1)

    n_class = pred_volume.shape[1]
    gt_1hot_volume = F.one_hot(segmentation_volume[:, 0], n_class).permute(0, 4, 1, 2, 3).float()

    scores = []
    for c in range(n_class):
        scores.append(loss_func(pred_volume[0, c], gt_1hot_volume[0, c]))

    if drop_bg_class:
        scores = scores[1:]

    return torch.stack(scores).mean()


def well_classified_voxel_perc(pred_volume, segmentation_volume):
    """
    :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
    :param segmentation_volume: uint8 array of shape (1, 1, slices, H, W) containing segmentation labels
    """
    assert (pred_volume.shape[0] == 1)

    n_class = pred_volume.shape[1]
    gt_1hot_volume = F.one_hot(segmentation_volume[:, 0], n_class).permute(0, 4, 1, 2, 3).float()

    intersection = torch.all(gt_1hot_volume == pred_volume, dim=1).sum()
    union = pred_volume.shape[-3:].numel()

    return intersection / union


def compute_segmentation_loss(pred_volume, segmentation_volume):
    """
    :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
    :param segm_map: uint8 array of shape (1, 1, slices, H, W) containing segmentation labels
    """
    pred_volume = F.softmax(pred_volume, dim=1).float()
    dice = per_class_score(pred_volume, segmentation_volume, loss_func=tversky_score)

    return 1 - dice


def compute_segmentation_score(pred_volume, segmentation_volume):
    """
    :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
    :param segm_map: uint8 array of shape (1, 1, slices, H, W) containing segmentation labels
    """
    # turn logits into one hot vector
    pred_map_volume = torch.argmax(pred_volume, dim=1, keepdim=True)
    pred_volume = F.one_hot(pred_map_volume[:, 0], pred_volume.shape[1]).permute(0, 4, 1, 2, 3)

    return per_class_score(pred_volume, segmentation_volume, loss_func=tversky_score)


if __name__ == '__main__':
    from evaluate import evaluate

    pred = torch.zeros(1, 3, 4, 5, 5).float()
    pred[:, 0] = 5
    gt = torch.zeros(1, 1, 4, 5, 5).long()

    # pred[0, :, 1, 1, 1] = torch.tensor([1,1,10])
    # gt[0, 0, 1, 1, 1] = 1
    #
    # pred[0, :, 3, 3, 3] = torch.tensor([1,1,10])
    # gt[0, 0, 3, 3, 3] = 2

    dice = compute_segmentation_score(pred, gt)
    dice_loss = compute_segmentation_loss(pred, gt)
    print(dice, dice_loss)
