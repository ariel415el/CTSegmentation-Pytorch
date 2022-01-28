import torch
from torch import nn
from metrics import compute_segmentation_loss, TverskyScore, SliceLoss
from models.Unet.model import UnetModel
from models.Unet2_5D.model import Unet2_5DModel
from torchvision.models import vgg13_bn


def load_vgg_weights(unet):
    # load vgg weights
    vgg = vgg13_bn(pretrained=True)
    unet.inc.double_conv.load_state_dict(vgg.features[:6].state_dict())
    for i in range(6):
        unet.down1.maxpool_conv[1].double_conv[i].load_state_dict(vgg.features[7 + i].state_dict())
        unet.down2.maxpool_conv[1].double_conv[i].load_state_dict(vgg.features[14 + i].state_dict())
        unet.down3.maxpool_conv[1].double_conv[i].load_state_dict(vgg.features[21 + i].state_dict())
        unet.down4.maxpool_conv[1].double_conv[i].load_state_dict(vgg.features[28 + i].state_dict())

class VGGUnet2_5DModel(Unet2_5DModel):
    def __init__(self, n_classes, lr, bilinear, eval_batchsize=1):
        super(VGGUnet2_5DModel, self).__init__(3, n_classes, lr, bilinear, bias=True, eval_batchsize=eval_batchsize)
        load_vgg_weights(self.net)

class VGGUnetModel(UnetModel):
    def __init__(self, n_classes, lr, bilinear, eval_batchsize=1):
        super(VGGUnetModel, self).__init__(3, n_classes, lr, bilinear, eval_batchsize=eval_batchsize, bias=True)
        load_vgg_weights(self.net)

    def train_one_sample(self, ct_volume, gt_volume, mask_volume, global_step):
        self.train()
        pred = self.net(ct_volume.repeat(1, 3, 1, 1))

        loss = SliceLoss(pred, gt_volume, mask_volume)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"Dice+CE_loss": loss.item()}

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
                pred_volume = self.net(ct_volume[i: i + self.eval_batchsize].repeat(1, 3, 1, 1))
                pred_volumes.append(pred_volume)
                i += self.eval_batchsize

        return torch.cat(pred_volumes).permute(1, 0, 2, 3).unsqueeze(0)



