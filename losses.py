import torch
import torch.nn.functional as F


class TverskyScore:
    def __init__(self, fp_weight=0.5, fn_weight=0.5):
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight

    def __call__(self, pred_mat, gt_map):
        """
        Prioritize FN weight in segmentation map prediction.
        if the weights are booth 0.5 you get exactly the dice score (f1)
        :param pred_mat: a matrix of shape (b, slices, H, W) with values in [0,1]
        :param gt_map: matrix of shape (b, slices, H, W) with values in [0,1]
        :return:
        """
        tp = (pred_mat * gt_map).sum([1, 2, 3])
        fp = (pred_mat * (1 - gt_map)).sum([1, 2, 3])
        fn = ((1 - pred_mat) * (gt_map)).sum([1, 2, 3])

        denominator = tp + fp * self.fp_weight + fn * self.fn_weight
        nwhere = gt_map.sum([1, 2, 3]) == 0
        denominator[nwhere] = tp[nwhere]

        score = (tp + 1e-6) / (denominator + 1e-6)
        score = torch.clip(score, 0, 1)
        return score

def compute_IOU(pred_mat, gt_map):
    """
    :param pred_mat: a matrix of shape (b, slices, H, W) with values in [0,1]
    :param gt_map: matrix of shape (b, slices, H, W) with values in [0,1]
    """
    intersection = (pred_mat * gt_map).sum([1, 2, 3])
    union = (pred_mat + gt_map).sum([1, 2, 3]) - intersection
    results = intersection / union
    results[union == 0] = 1

    return results

def per_class_score(pred_volume, segmentation_volume, score_func, drop_bg_class=False):
    """
    :param pred_volume: float array of shape (b, n_class, slices, H, W) contating class logits
    :param segmentation_volume: uint8 array of shape (b, 1, slices, H, W) containing segmentation labels
    """
    n_class = pred_volume.shape[1]
    gt_1hot_volume = F.one_hot(segmentation_volume[:, 0], n_class).permute(0, 4, 1, 2, 3).float()

    scores = []
    for c in range(n_class):
        scores.append(score_func(pred_volume[:, c], gt_1hot_volume[:, c]))

    if drop_bg_class:
        scores = scores[1:]

    return torch.stack(scores).mean()

def compute_segmentation_loss(pred_volume, segmentation_volume, score_func):
    """
    :param pred_volume: float array of shape (b, n_class, slices, H, W) contating class logits
    :param segmentation_volume: uint8 array of shape (b, 1, slices, H, W) containing segmentation labels
    """
    pred_volume = F.softmax(pred_volume, dim=1).float()
    score = per_class_score(pred_volume, segmentation_volume, score_func=score_func)

    return (1 - score).mean()


def compute_segmentation_score(pred_volume, segmentation_volume, score_func):
    """
    :param pred_volume: float array of shape (b, n_class, slices, H, W) contating class logits
    :param segm_map: uint8 array of shape (b, 1, slices, H, W) containing segmentation labels
    """
    # turn logits into one hot vector
    pred_map_volume = torch.argmax(pred_volume, dim=1, keepdim=True)
    pred_volume = F.one_hot(pred_map_volume[:, 0], pred_volume.shape[1]).permute(0, 4, 1, 2, 3)

    score = per_class_score(pred_volume, segmentation_volume, score_func=score_func)
    return score.mean()

# def well_classified_voxel_perc(pred_volume, segmentation_volume):
#     """
#     :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
#     :param segmentation_volume: uint8 array of shape (1, 1, slices, H, W) containing segmentation labels
#     """
#     assert (pred_volume.shape[0] == 1)
#
#     n_class = pred_volume.shape[1]
#     gt_1hot_volume = F.one_hot(segmentation_volume[:, 0], n_class).permute(0, 4, 1, 2, 3).float()
#
#     intersection = torch.all(gt_1hot_volume == pred_volume, dim=1).sum()
#     union = pred_volume.shape[-3:].numel()
#
#     return intersection / union


