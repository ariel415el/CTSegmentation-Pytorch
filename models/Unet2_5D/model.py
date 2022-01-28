import torch
from torch import nn, optim
from metrics import compute_segmentation_loss, TverskyScore, SliceLoss
from models.Unet.net import UNet
from models.generic_model import SegmentationModel


class Unet2_5DModel(SegmentationModel):
    def __init__(self, slice_size, n_classes, lr, bilinear, bias=False, eval_batchsize=1):
        super(Unet2_5DModel, self).__init__(slice_size, n_classes)
        # assert slice_size % 2 == 1, "slice size  should be odd"
        assert slice_size == 3, "Currently only slice size=3 is suppported "
        self.net = UNet(slice_size, n_classes, bilinear=bilinear, bias=bias)
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5)  # goal: maximize val Dice score
        self.eval_batchsize = eval_batchsize

    def train_one_sample(self, ct_volume, gt_volume, mask_volume, global_step):
        B, S, H, W = ct_volume.shape
        m = 1
        self.train()
        middle_gt = gt_volume[:, m:m+1]
        middle_mask = mask_volume[:, m:m+1]
        middle_pred = self.net(ct_volume)

        loss = SliceLoss(middle_pred, middle_gt, middle_mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"Dice+CE_loss": loss.item()}

    def step_scheduler(self, evaluation_score):
        self.scheduler.step(evaluation_score)

    def predict_volume(self, ct_volume):
        """
        ct_volume.shape = (1, slices, H, W)
        returns prdiction of shape (1, n_classes, slices, H, W)
        """
        self.net.eval()
        B, S, H, W = ct_volume.shape
        m=1
        # padd
        ct_volume = torch.cat([ct_volume[:, :m], ct_volume, ct_volume[:, -m:]], dim=1)
        pred_volumes = []
        with torch.no_grad():
            for i in range(1,S+1):
                ct_slice = ct_volume[:, i-m:i+m+1]
                pred_volume = self.net(ct_slice)
                pred_volumes.append(pred_volume)

        return torch.cat(pred_volumes).permute(1, 0, 2, 3).unsqueeze(0)

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