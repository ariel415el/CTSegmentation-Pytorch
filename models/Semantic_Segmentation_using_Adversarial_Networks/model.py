import numpy as np
import torch

from models.Semantic_Segmentation_using_Adversarial_Networks.segmentor import fcn32s
from models.Semantic_Segmentation_using_Adversarial_Networks.discriminator import StanfordBNet
from dice_score import compute_segmentation_loss
from torch import optim
import torch.nn.functional as F

def VolumeLoss(preds, gts):
    """
    :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
    :param gts: uint8 array of shape (1, slices, H, W) containing segmentation labels
    """
    dice_loss = compute_segmentation_loss(preds, gts.unsqueeze(1))
    return dice_loss



class AdSegModel:
    def __init__(self, n_channels, n_classes, lr=1e-5, device=torch.device('cpu')):
        self.n_classes = n_classes
        self.segmentor = fcn32s(n_channels, n_classes).to(device)
        self.discriminator = StanfordBNet(n_channels, n_classes).to(device)
        self.discriminator.train()
        self.s_optimizer = optim.Adam(self.segmentor.parameters(), lr=lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.bce_loss = torch.nn.BCELoss(reduction='sum')
        self.ce_loss = torch.nn.CrossEntropyLoss()
        # self.d_steps = d_steps

    def _discriminator_loss(self, d_out_on_gt, d_out_on_pred):
        batch_size = d_out_on_gt.shape[0]
        device = d_out_on_gt.device
        zeros = torch.zeros(batch_size, requires_grad=False).to(device)
        ones = torch.ones(batch_size, requires_grad=False).to(device)
        gt_loss = self.bce_loss(d_out_on_gt, ones)
        pred_loss = self.bce_loss(d_out_on_pred, zeros)

        return gt_loss, pred_loss

    def _segmentor_adverserial_loss(self, d_out_on_pred):
        batch_size = d_out_on_pred.shape[0]
        device = d_out_on_pred.device
        ones = torch.ones(batch_size, requires_grad=False).to(device)
        adverserial_loss = self.bce_loss(d_out_on_pred, ones)

        return adverserial_loss

    def train_one_sample(self, ct_volume, gt_volume, global_step):
        self.segmentor.train()

        # Train discriminator:
        self.d_optimizer.zero_grad()

        pred_volume = self.segmentor(ct_volume)
        d_outputs_pred = self.discriminator(pred_volume.detach(), ct_volume)
        gt_one_hot = F.one_hot(gt_volume[:, 0], 3).permute(0, 3, 1, 2).float()
        d_outputs_gt = self.discriminator(gt_one_hot, ct_volume)

        d_loss_gt, d_loss_pred = self._discriminator_loss(d_outputs_gt, d_outputs_pred)
        d_loss_gt.backward()
        d_loss_pred.backward()

        self.d_optimizer.step()

        # Train segmentor
        # if global_step % self.d_steps == 0:
        self.s_optimizer.zero_grad()

        ce_loss = self.ce_loss(pred_volume, gt_volume[:, 0])
        d_outputs_pred = self.discriminator(pred_volume, ct_volume) # Can we avoid repeated inference here?
        adverserial_loss = self._segmentor_adverserial_loss(d_outputs_pred)
        g_loss = ce_loss + 0.65 * adverserial_loss
        g_loss.backward()

        self.s_optimizer.step()

        return ce_loss.item()

    def step_scheduler(self, evaluation_score):
        pass

    def predict_volume(self, ct_volume):
        """
        ct_volume.shape = (b, slices, H, W)
        returns prdiction of shape (b, n_classes, slices, H, W)
        """
        self.segmentor.eval()
        pred_volume = []
        with torch.no_grad():
            for i in range(ct_volume.shape[1]):
                image = ct_volume[:, i]
                pred_volume.append(self.segmentor(image.unsqueeze(1)))
        pred_volume = torch.stack(pred_volume, dim=2)
        return pred_volume

    def get_state_dict(self):
        return {
            'discriminator': self.discriminator.state_dict(),
            'segmentor': self.segmentor.state_dict(),
            's_optimizer': self.s_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.discriminator.load_state_dict(state_dict['discriminator'])
        self.segmentor.load_state_dict(state_dict['segmentor'])
        self.s_optimizer.load_state_dict(state_dict['s_optimizer'])
        self.d_optimizer.load_state_dict(state_dict['d_optimizer'])
