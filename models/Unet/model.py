import torch
from torch import nn, optim
from models.Unet.net import UNet
from metrics import VolumeLoss
from models.generic_model import SegmentationModel, optimizer_to


class UnetModel(SegmentationModel):
    def __init__(self, n_channels, n_classes, lr, bilinear=True, bias=False, eval_batchsize=1):
        super(UnetModel, self).__init__(n_channels, n_classes)
        self.net = UNet(n_channels, n_classes, bilinear=bilinear, bias=bias)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.eval_batchsize = eval_batchsize

    def train_one_sample(self, ct_volume, gt_volume, mask_volume):
        self.net.train()
        pred = self.net(ct_volume)

        loss = VolumeLoss(pred.unsqueeze(2), gt_volume, mask_volume)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict_volume(self, ct_volume):
        """
        ct_volume.shape = (1, slices, H, W)
        returns prdiction of shape (1, n_classes, slices, H, W)
        """
        self.net.eval()
        _, S, H, W = ct_volume.shape
        pred_volumes = []
        ct_volume = ct_volume.view(S, 1, H, W)
        with torch.no_grad():
            i = 0
            while i < S:
                pred_volume = self.net(ct_volume[i: i + self.eval_batchsize])
                pred_volumes.append(pred_volume)
                i += self.eval_batchsize

        return torch.cat(pred_volumes).permute(1, 0, 2, 3).unsqueeze(0)

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