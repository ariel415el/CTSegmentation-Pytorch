import torch

from models.DARN.net import DARN
from metrics import VolumeLoss
from torch import optim

from models.generic_model import SegmentationModel, optimizer_to


class DARNModel(SegmentationModel):
    def __init__(self, n_channels, n_classes, slice_size, lr):
        super(DARNModel, self).__init__(n_channels, n_classes)
        self.net = DARN(n_channels, n_classes)
        self.slice_size = slice_size
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def train_one_sample(self, ct_volume, gt_volume, mask_volume):
        self.net.train()
        pred, intermediate_maps = self.net(ct_volume)

        # deep_loss = multipleVolumeSLoss(intermediate_maps, gt_volume, mask_volume)
        loss = VolumeLoss(pred, gt_volume, mask_volume)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

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

    def decay_learning_rate(self, factor):
        for g in self.optimizer.param_groups:
            g['lr'] *= factor

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
        optimizer_to(self.optimizer, device)