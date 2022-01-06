import torch

from models.Vnet.net import VNet
from dice_score import compute_segmentation_loss
from torch import optim


def VolumeLoss(preds, gts):
    """
    :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
    :param gts: uint8 array of shape (1, slices, H, W) containing segmentation labels
    """
    dice_loss = compute_segmentation_loss(preds, gts.unsqueeze(1))
    return dice_loss


class VnetModel:
    def __init__(self, n_channels, n_classes, d=16, lr=0.001, device=torch.device('cpu')):
        self.n_classes = n_classes
        self.net = VNet(n_channels, n_classes, d=d).to(device)
        self.d = d
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2)  # goal: maximize val Dice score

    def train_one_sample(self, ct_volume, gt_volume, global_step):
        self.net.train()
        pred = self.net(ct_volume)

        loss = VolumeLoss(pred, gt_volume)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def step_scheduler(self, evaluation_score):
        self.scheduler.step(evaluation_score)

    def predict_volume(self, ct_volume, overlap=None):
        """
        ct_volume.shape = (b, slices, H, W)
        returns prdiction of shape (b, n_classes, slices, H, W)
        """
        self.net.eval()
        if not overlap:
            overlap = self.d // 2
        pred_volume = torch.stack([torch.zeros_like(ct_volume)] * 3, dim=1).to(device=ct_volume.device)
        pred_counters = torch.stack([torch.zeros_like(ct_volume)] * 3, dim=1).to(device=ct_volume.device)
        with torch.no_grad():
            for i in range(0, ct_volume.shape[-3] - self.d + 1, overlap):
                pred_volume[..., i: i + self.d, :, :] += self.net(ct_volume[..., i: i + self.d, :, :])
                pred_counters[..., i: i + self.d, :, :] += 1
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