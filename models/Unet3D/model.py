import torch

from metrics import VolumeLoss
from models.Unet3D.net import UNet3D, UNet3D_new
from torch import optim

from models.generic_model import SegmentationModel, optimizer_to


class UNet3DModel(SegmentationModel):
    def __init__(self, n_classes, trilinear, slice_size=16, lr=0.0001):
        super(UNet3DModel, self).__init__(1, n_classes)
        self.net = UNet3D(1, n_classes, trilinear=trilinear)
        self.slice_size = slice_size
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def train_one_sample(self, ct_volume, gt_volume, mask_volume):
        self.net.train()
        pred = self.net(ct_volume)

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
        optimizer_to(self.optimizer, device)