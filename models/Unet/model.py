import torch
from torch import nn
from models.Unet.net import UNet
from dice_score import compute_segmentation_loss


def SliceLoss(preds, gts):
    """
    :param preds: float array of shape (1, n_class, H, W) contating class logits
    :param gts: uint8 array of shape (1, 1, H, W) containing segmentation labels
    """
    ce_loss = nn.CrossEntropyLoss()(preds, gts[:,0])
    dice_loss = compute_segmentation_loss(preds, gts.unsqueeze(1))
    return ce_loss + dice_loss


class UnetModel:
    def __init__(self, n_channels, n_classes, bilinear=True):
        self.net = UNet(n_channels, n_classes, bilinear=bilinear)

    def predict_volume(self, ct_volume):
        """
        ct_volume.shape = (b, slices, H, W)
        returns prdiction of shape (b, n_classes, slices, H, W)
        """
        pred_volume = []
        with torch.no_grad():
            for i in range(ct_volume.shape[1]):
                image = ct_volume[:, i]
                pred_volume.append(self.net(image.unsqueeze(1)))
        pred_volume = torch.stack(pred_volume, dim=2)
        return pred_volume


