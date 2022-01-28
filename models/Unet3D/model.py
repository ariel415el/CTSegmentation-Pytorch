import torch

from models.Unet3D.net import UNet3D
from metrics import compute_segmentation_loss, TverskyScore
from torch import optim

from models.generic_model import SegmentationModel


def VolumeLoss(preds, gts, mask):
    """
    :param preds: float array of shape (b, n_class, slices, H, W) contating class logits
    :param gts: uint8 array of shape (b, slices, H, W) containing segmentation labels
    :param mask: bool array of shape (b, slices, H, W) containing segmentation labels
    """
    dice_loss = compute_segmentation_loss(TverskyScore(0.5, 0.5), preds, gts.unsqueeze(1), mask.unsqueeze(1))
    return dice_loss


class UNet3DModel(SegmentationModel):
    def __init__(self, n_channels, n_classes, slice_size=16, lr=0.0001):
        super(UNet3DModel, self).__init__(n_channels, n_classes)
        self.net = UNet3D(n_channels, n_classes)
        self.slice_size = slice_size
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2)  # goal: maximize val Dice score

    def train_one_sample(self, ct_volume, gt_volume, mask_volume, global_step):
        self.net.train()
        pred = self.net(ct_volume)

        loss = VolumeLoss(pred, gt_volume, mask_volume)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"Dice_loss": loss.item()}

    def step_scheduler(self, evaluation_score):
        self.scheduler.step(evaluation_score)
        # for g in self.optimizer.param_groups:
        #     g['lr'] *= 0.95

    def predict_volume(self, ct_volume, overlap=None):
        """
        ct_volume.shape = (1, slices, H, W)
        returns prdiction of shape (1, n_classes, slices, H, W)
        """
        self.net.eval()
        if not overlap:
            overlap = self.slice_size
        pred_volume = torch.stack([torch.zeros_like(ct_volume)] * self.n_classes, dim=1).to(device=ct_volume.device)
        pred_counters = torch.stack([torch.zeros_like(ct_volume)] * self.n_classes, dim=1).to(device=ct_volume.device)
        with torch.no_grad():
            for i in range(0, ct_volume.shape[-3] - self.slice_size, overlap):
                pred_volume[..., i: i + self.slice_size, :, :] += self.net(ct_volume[..., i: i + self.slice_size, :, :])
                pred_counters[..., i: i + self.slice_size, :, :] += 1
        if i < ct_volume.shape[-3] - overlap - 1:
            pred_volume[..., -overlap:, :, :] += self.net(ct_volume[..., -overlap:, :, :])
            pred_counters[..., -overlap:, :, :] += 1

        nwhere = pred_counters != 0
        pred_volume[nwhere] /= pred_counters[nwhere]
        return pred_volume

    def get_state_dict(self):
        return {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def to(self, device):
        self.net.to(device=device)