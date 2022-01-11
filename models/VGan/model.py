import torch

from torch import optim
import torch.nn.functional as F
from torch.autograd import grad

from models.VGan.networks import VGanDiscriminator
from models.VGan.networks import VGanGenerator
from models.generic_model import SegmentationModel


class VGanModel(SegmentationModel):
    def __init__(self, n_channels, n_classes, device=torch.device('cpu'), eval_batchsize=1):
        super(VGanModel, self).__init__(n_channels, n_classes, device)
        self.eval_batchsize = eval_batchsize
        self.segmentor = VGanGenerator(n_channels, n_classes).to(device)
        self.discriminator = VGanDiscriminator(n_channels + n_classes).to(device)
        self.discriminator.train()
        self.s_optimizer = optim.Adam(self.segmentor.parameters(), lr=1e-4,betas=(0.5,0.9),eps=10e-8)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4,betas=(0.5,0.9),eps=10e-8)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.d_steps = 10

    def _calc_gradient_penalty(self, netD, real_data, fake_data, LAMBDA=10):
        BATCH = real_data.size()[0]
        alpha = torch.rand(BATCH, 1)
        # print(alpha.size(),real_data.size())
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).expand(real_data.size())
        alpha = alpha.cuda()
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda()
        interpolates.requires_grad=True

        disc_interpolates = netD(interpolates)

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def train_one_sample(self, ct_volume, gt_volume, global_step):
        self.segmentor.train()

        # Train discriminator:
        self.d_optimizer.zero_grad()

        gt_one_hot = F.one_hot(gt_volume[:, 0], self.n_classes).permute(0, 3, 1, 2).float()
        real_pair = torch.cat([ct_volume, gt_one_hot], dim=1)
        d_outputs_gt = self.discriminator(real_pair).mean()

        pred_volume = self.segmentor(ct_volume)
        fake_pair = torch.cat([ct_volume, pred_volume.detach()], dim=1)
        d_outputs_pred = self.discriminator(fake_pair).mean()

        gradient_penalty=self._calc_gradient_penalty(self.discriminator, real_pair.data, fake_pair.data)

        loss = d_outputs_gt - d_outputs_pred + gradient_penalty
        loss.backward()
        self.d_optimizer.step()

        losses = {"d_gt": d_outputs_gt.item(), "d_pred": d_outputs_pred.item(),
                  "gradient_penalty": gradient_penalty.item()}

        # Train segmentor
        if global_step % self.d_steps == 0:
            self.segmentor.zero_grad()
            pred_volume = self.segmentor(ct_volume)
            ce_loss = self.ce_loss(pred_volume, gt_volume[:, 0])  # Seg Loss
            fake_pair = torch.cat([ct_volume, pred_volume], dim=1)
            adverserial_loss = self.discriminator(fake_pair).mean()
            g_loss = ce_loss -0.03 * adverserial_loss
            g_loss.backward()
            self.s_optimizer.step()

            losses["ce_loss"] = ce_loss.item()
            losses["adv_loss"] = adverserial_loss.item()

        return losses

    def step_scheduler(self, evaluation_score):
        pass

    def predict_volume(self, ct_volume):
        """
        ct_volume.shape = (1, slices, H, W)
        returns prdiction of shape (1, n_classes, slices, H, W)
        """
        self.segmentor.eval()
        _, S, H, W = ct_volume.shape
        pred_volumes = []
        ct_volume = ct_volume.view(S, 1, H, W)
        with torch.no_grad():
            i = 0
            while i < S:
                pred_volume = self.segmentor(ct_volume[i: i + self.eval_batchsize])
                pred_volumes.append(pred_volume)
                i += self.eval_batchsize

        return torch.cat(pred_volumes).permute(1, 0, 2, 3).unsqueeze(0)

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