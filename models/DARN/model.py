import torch

from models.DARN.net import DARN
from metrics import compute_segmentation_loss, TverskyScore
from torch import optim

from models.generic_model import SegmentationModel


def multipleVolumeSLoss(pred_maps, gts, mask):
    """
    :param preds: list of float arrays of shape (b, n_class, slices, H, W) contating class logits
    :param gts: uint8 array of shape (b, slices, H, W) containing segmentation labels
    :param mask: bool array of shape (b, slices, H, W) containing segmentation labels
    """
    dice_loss = 0
    for preds in pred_maps:
        dice_loss += compute_segmentation_loss(TverskyScore(0.5, 0.5), preds, gts.unsqueeze(1), mask.unsqueeze(1))
    return dice_loss


class DARNModel(SegmentationModel):
    def __init__(self, n_channels, n_classes, slice_size, lr, device):
        super(DARNModel, self).__init__(n_channels, n_classes, device)
        self.net = DARN(n_channels, n_classes).to(device)
        self.slice_size = slice_size
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2)  # goal: maximize val Dice score

    def train_one_sample(self, ct_volume, gt_volume, mask_volume, global_step):
        self.net.train()
        _, intermediate_maps = self.net(ct_volume)

        loss = multipleVolumeSLoss(intermediate_maps, gt_volume, mask_volume)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"Dice_loss": loss.item()}

    def step_scheduler(self, evaluation_score):
        self.scheduler.step(evaluation_score)

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
                pred_slices, _ = self.net(ct_volume[..., i: i + self.slice_size, :, :])
                pred_volume[..., i: i + self.slice_size, :, :] += pred_slices
                pred_counters[..., i: i + self.slice_size, :, :] += 1
        if i < ct_volume.shape[-3] - overlap - 1:
            pred_slices, _ = self.net(ct_volume[..., -overlap:, :, :])
            pred_volume[..., -overlap:, :, :] += pred_slices
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