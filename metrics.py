import torch
import torch.nn.functional as F


class TverskyScore:
    def __init__(self, fp_weight=0.5, fn_weight=0.5):
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight

    def __call__(self, pred_mat, gt_map, mask):
        """
        Prioritize FN weight in segmentation map prediction.
        if the weights are booth 0.5 you get exactly the dice score (f1)
        :param pred_mat: a matrix of shape (b, slices, H, W) with values in [0,1]
        :param gt_map: matrix of shape (b, slices, H, W) with values in [0,1]
        :param mask: boolean matrix of shape (slices, H, W)
        :return: scores array of size b
        """

        pred_mat = pred_mat * mask
        gt_map = gt_map * mask

        tp = (pred_mat * gt_map).sum([1,2,3])
        fp = (pred_mat * (1 - gt_map)).sum([1,2,3])
        fn = ((1 - pred_mat) * (gt_map)).sum([1,2,3])

        denominator = tp + fp * self.fp_weight + fn * self.fn_weight

        # # perfect score on epty
        # nwhere = gt_map.sum([1,2,3]) == 0
        # if torch.any(nwhere):
        #     denominator[nwhere] = tp[nwhere]

        score = (tp + 1e-6) / (denominator + 1e-6)
        score = torch.clip(score, 0, 1)
        return score


def compute_IOU(pred_mat, gt_map, mask):
    """
    :param pred_mat: a matrix of shape (b, slices, H, W) with values in [0,1]
    :param gt_map: matrix of shape (b, slices, H, W) with values in [0,1]
    :param mask: boolean matrix of shape (b, slices, H, W)
    :return: scores array of size b
    """
    pred_mat *= mask
    gt_map *= mask

    intersection = (pred_mat * gt_map).sum([1,2,3])
    union = (pred_mat + gt_map).sum([1,2,3]) - intersection
    results = intersection / union
    results[union == 0] = 1

    return results

def compute_Recal(pred_mat, gt_map, mask):
    """
    :param pred_mat: a matrix of shape (b, slices, H, W) with values in [0,1]
    :param gt_map: matrix of shape (b, slices, H, W) with values in [0,1]
    :param mask: boolean matrix of shape (b, slices, H, W)
    :return: scores array of size b
    """
    pred_mat *= mask
    gt_map *= mask

    tp = (pred_mat * gt_map).sum([1,2,3])
    T = (gt_map).sum([1,2,3])
    results = tp / T
    results[T == 0] = 1

    return results



def per_class_score(score_func, pred_volume, segmentation_volume, mask_volume=None):
    """
    :param pred_volume: float array of shape (b, n_class, slices, H, W) contating class logits
    :param segmentation_volume: uint8 array of shape (b, 1, slices, H, W) containing segmentation labels
    :param mask_volume: (optional) bool array of shape (b, 1, H, W) indicating relevant voxels

    """
    if mask_volume is None:
        mask_volume = torch.ones_like(segmentation_volume).to(segmentation_volume.device)

    n_class = pred_volume.shape[1]
    gt_1hot_volume = F.one_hot(segmentation_volume[:, 0], n_class).permute(0, 4, 1, 2, 3).float()


    scores = []
    for c in range(n_class):
        scores.append(score_func(pred_volume[:, c], gt_1hot_volume[:, c],  mask_volume[:, 0]))

    return torch.stack(scores).mean(1)


def compute_segmentation_loss(score_func, pred_volume, segmentation_volume, mask_volume=None):
    """
    :param pred_volume: float array of shape (b, n_class, slices, H, W) contating class logits
    :param segmentation_volume: uint8 array of shape (b, 1, slices, H, W) containing segmentation labels
    :param mask_volume: bool array of shape (b, 1, slices, H, W) indicating relevant voxels

    """
    pred_volume = F.softmax(pred_volume, dim=1).float()

    score = per_class_score(score_func, pred_volume, segmentation_volume, mask_volume)

    loss = 1 - score
    loss = loss[1:].mean() # ignore background class

    return loss


def compute_segmentation_score(score_func, pred_volume, segmentation_volume, mask_volume=None, return_per_class=False):
    """
    :param pred_volume: float array of shape (b, n_class, slices, H, W) contating class logits
    :param segm_map: uint8 array of shape (b, 1, slices, H, W) containing segmentation labels
    :param mask_volume: bool array of shape (b, 1, slices, H, W) indicating relevant voxels
    """

    # turn logits into one hot vector
    pred_map_volume = torch.argmax(pred_volume, dim=1, keepdim=True)

    pred_volume = F.one_hot(pred_map_volume[:, 0], pred_volume.shape[1]).permute(0, 4, 1, 2, 3)

    scores = per_class_score(score_func, pred_volume, segmentation_volume, mask_volume=mask_volume)
    if return_per_class:
        return scores
    else:
        return scores.mean()


class VolumeLoss:
    def __init__(self, dice_weight, wce_weight, ce_weight):
        self.dice_weight = dice_weight
        self.wce_weight = wce_weight
        self.ce_weight = ce_weight
        self.dice_loss = TverskyScore(0.5, 0.5)

    def __str__(self):
        return f"Vloss({self.dice_weight:.1f}Dice+{self.ce_weight:.1f}WCE)"

    def __call__(self, preds, gts, mask):
        """
        :param preds: float array of shape (b, n_class, slices, H, W) contating class logits
        :param gts: uint8 array of shape (b, slices, H, W) containing segmentation labels
        :param mask: bool array of shape (b, slices, H, W) containing segmentation labels
        """
        loss = 0
        if self.dice_weight > 0:
            dice_loss = compute_segmentation_loss(self.dice_loss, preds, gts.unsqueeze(1), mask.unsqueeze(1))
            loss += self.dice_weight * dice_loss
        if self.wce_weight > 0:
            class_weights = torch.tensor([1,(gts[mask] == 0).sum() / (gts[mask] == 1).sum()]).to(preds.device)
            wce_loss = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')(preds, gts)[mask].mean()
            loss += self.wce_weight * wce_loss
        elif self.ce_weight > 0:
            ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(preds, gts)[mask].mean()
            loss += self.ce_weight * ce_loss
        return loss


if __name__ == '__main__':
    import numpy as np
    from medpy.metric import dc
    gt = np.random.randint(0, 2, (5,128,128))
    pred = np.random.randint(0, 2, (5,128,128))
    gt = np.zeros((5,128,128))

    score = dc(pred, gt)
    print(score)

    score = TverskyScore(0.5,0.5)(torch.from_numpy(pred)[None, :], torch.from_numpy(gt)[None, :], mask=torch.ones(1,5,128,128).bool())
    print(score)