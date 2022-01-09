import torch

from models.Adversarial_Learning_For_Semi_Supervised_Semantic_Segmentation.discriminator import Dis
from models.Adversarial_Learning_For_Semi_Supervised_Semantic_Segmentation.segmentor import ResDeeplab
from torch import optim
import torch.nn.functional as F

from models.generic_model import SegmentationModel


class AdverserialSegSemi(SegmentationModel):
    def __init__(self, n_channels, n_classes, device=torch.device('cpu')):
        super(AdverserialSegSemi, self).__init__(n_channels, n_classes, device)
        self.segmentor = ResDeeplab(n_channels, n_classes).to(device)
        self.discriminator = Dis(n_classes).to(device)
        self.discriminator.train()
        self.s_optimizer = optim.Adam(self.segmentor.parameters(), lr=0.0001)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.00025)
        self.bce_loss = torch.nn.BCELoss(reduction='sum')
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def _discriminator_loss(self, d_out_on_gt, d_out_on_pred):
        device = d_out_on_gt.device
        zeros = torch.zeros_like(d_out_on_pred, requires_grad=False).to(device)
        ones = torch.ones_like(d_out_on_gt, requires_grad=False).to(device)
        gt_loss = self.bce_loss(d_out_on_gt, ones)
        pred_loss = self.bce_loss(d_out_on_pred, zeros)

        return gt_loss, pred_loss

    def _segmentor_adverserial_loss(self, d_out_on_pred):
        device = d_out_on_pred.device
        ones = torch.ones_like(d_out_on_pred, requires_grad=False).to(device)
        adverserial_loss = self.bce_loss(d_out_on_pred, ones)

        return adverserial_loss

    def train_one_sample(self, ct_volume, gt_volume, global_step):
        self.segmentor.train()

        # Train discriminator:
        self.d_optimizer.zero_grad()

        pred_volume = self.segmentor(ct_volume)
        d_outputs_pred = self.discriminator(pred_volume.detach())
        gt_one_hot = F.one_hot(gt_volume[:, 0], 3).permute(0, 3, 1, 2).float()
        d_outputs_gt = self.discriminator(gt_one_hot)

        d_loss_gt, d_loss_pred = self._discriminator_loss(d_outputs_gt, d_outputs_pred)
        d_loss_gt.backward()
        d_loss_pred.backward()

        self.d_optimizer.step()

        # Train segmentor
        # if global_step % self.d_steps == 0:
        self.s_optimizer.zero_grad()

        ce_loss = self.ce_loss(pred_volume, gt_volume[:, 0])
        d_outputs_pred = self.discriminator(pred_volume) # Can we avoid repeated inference here?
        adverserial_loss = self._segmentor_adverserial_loss(d_outputs_pred)
        g_loss = ce_loss + 0.01 * adverserial_loss
        g_loss.backward()

        self.s_optimizer.step()

        return {"ce_loss": ce_loss.item(), 'G_adv_loss':adverserial_loss.item(),
                "d_loss_gt": d_loss_gt.item(), "d_loss_pred": d_loss_pred.item()}

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

    def train(self):
        self.segmentor.train()
        self.discriminator.train()

    def eval(self):
        self.segmentor.eval()
        self.discriminator.eval()