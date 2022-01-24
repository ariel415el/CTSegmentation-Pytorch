import torch
from torch import nn, optim
from models.Unet.net import UNet
from metrics import compute_segmentation_loss, TverskyScore
from models.generic_model import SegmentationModel


def SliceLoss(preds, gts, mask):
    """
    :param preds: float array of shape (b, n_class, H, W) contating class logits
    :param gts: uint8 array of shape (b, 1, H, W) containing segmentation labels
    :param mask: bool array of shape (b, 1, H, W) containing segmentation labels
    """
    dice_loss = compute_segmentation_loss(TverskyScore(0.5, 0.5), preds.unsqueeze(2), gts.unsqueeze(2), mask.unsqueeze(2))
    # ce_loss = nn.CrossEntropyLoss()(preds, gts[:, 0])
    class_weights = torch.tensor([1,(gts == 0).sum() / (gts == 1).sum()]).to(preds.device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)(preds, gts[:, 0])
    # return dice_loss
    return ce_loss + dice_loss


class UnetModel(SegmentationModel):
    def __init__(self, n_channels, n_classes, lr, bilinear=True, device=torch.device('cpu'), eval_batchsize=1):
        super(UnetModel, self).__init__(n_channels, n_classes, device)
        self.net = UNet(n_channels, n_classes, bilinear=bilinear).to(device)
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        # self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=0.0005, momentum=0.8)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2)  # goal: maximize val Dice score
        self.eval_batchsize = eval_batchsize

    def train_one_sample(self, ct_volume, gt_volume, mask_volume, global_step):
        self.net.train()
        pred = self.net(ct_volume)

        loss = SliceLoss(pred, gt_volume, mask_volume)

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
