import torch
from torch import optim

from models.Res2UNet.net import U_Net, ResU_Net, RecU_Net, R2U_Net
from models.generic_model import SegmentationModel, optimizer_to


class HeavyUnetModel(SegmentationModel):
    def __init__(self, n_channels, n_classes, lr, eval_batchsize=1):
        super(HeavyUnetModel, self).__init__(n_channels, n_classes)
        self.net = U_Net(n_channels, n_classes)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.eval_batchsize = eval_batchsize
        self.name = f"HeavyUNet"

    def __str__(self):
        return self.name

    def train_one_sample(self, ct_volume, gt_volume, mask_volume, volume_crieteria):
        self.net.train()
        pred = self.net(ct_volume)

        loss = volume_crieteria(pred.unsqueeze(2), gt_volume, mask_volume)

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

class ResUnetModel(HeavyUnetModel):
    def __init__(self, n_channels, n_classes, lr, eval_batchsize=1):
        super(ResUnetModel, self).__init__(n_channels, n_classes, lr, eval_batchsize=eval_batchsize)
        self.net = ResU_Net(n_channels, n_classes)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.name = f"ResUNet"


class RecurrentUnetModel(HeavyUnetModel):
    def __init__(self, n_channels, n_classes, lr, eval_batchsize=1):
        super(RecurrentUnetModel, self).__init__(n_channels, n_classes, lr, eval_batchsize=eval_batchsize)
        self.net = RecU_Net(n_channels, n_classes)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.name = f"RecurrentUNet"


class Res2UnetModel(HeavyUnetModel):
    def __init__(self, n_channels, n_classes, lr, eval_batchsize=1):
        super(Res2UnetModel, self).__init__(n_channels, n_classes, lr, eval_batchsize=eval_batchsize)
        self.net = R2U_Net(n_channels, n_classes)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.name = f"Res2Unet"