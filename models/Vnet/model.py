import torch

from models.Vnet.net import VNet
from dice_score import compute_segmentation_loss


def VolumeLoss(preds, gts):
    """
    :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
    :param gts: uint8 array of shape (1, slices, H, W) containing segmentation labels
    """
    dice_loss = compute_segmentation_loss(preds, gts.unsqueeze(1))
    return dice_loss


class VnetModel:
    def __init__(self, n_channels, n_classes, d=16):
        self.net = VNet(n_channels, n_classes, d=d)
        self.d = d

    def predict_volume(self, ct_volume, overlap=None):
        """
        ct_volume.shape = (b, slices, H, W)
        returns prdiction of shape (b, n_classes, slices, H, W)
        """
        if not overlap:
            overlap = self.d // 2
        pred_volume = torch.stack([torch.zeros_like(ct_volume)] * 3, dim=1).to(device=ct_volume.device)
        pred_counters = torch.stack([torch.zeros_like(ct_volume)] * 3, dim=1).to(device=ct_volume.device)
        with torch.no_grad():
            for i in range(0, ct_volume.shape[-3] - self.d + 1, overlap):
                pred_volume[..., i: i + self.d, :, :] += self.net(ct_volume[..., i: i + self.d, :, :])
                pred_counters[..., i: i + self.d, :, :] += 1
        pred_volume /= pred_counters
        return pred_volume
