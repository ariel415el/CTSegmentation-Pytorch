""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Unet.net import UNet


class VGGUNet(UNet):
    def __init__(self, n_classes, bilinear=True):
        super(VGGUNet, self).__init__(3, n_classes, bilinear=bilinear, bias=True)
        self.load_weigts()

    def load_weigts(self):
        from torchvision.models import vgg13_bn
        vgg = vgg13_bn(pretrained=True)
        self.inc.double_conv.load_state_dict(vgg.features[:6].state_dict())
        for i in range(6):
            self.down1.maxpool_conv[1].double_conv[i].load_state_dict(vgg.features[7 + i].state_dict())
            self.down2.maxpool_conv[1].double_conv[i].load_state_dict(vgg.features[14 + i].state_dict())
            self.down3.maxpool_conv[1].double_conv[i].load_state_dict(vgg.features[21 + i].state_dict())
            self.down4.maxpool_conv[1].double_conv[i].load_state_dict(vgg.features[28 + i].state_dict())



if __name__ == '__main__':
    net = UNet(3,2)
    net.eval()
    x1 = torch.zeros((1,3,16,16))
    x2 = torch.zeros((1,3,16,16))

    input_batch = torch.cat([x1,x2], dim=0)
    input_tile_hor = torch.cat([x1,x2], dim=-1)
    input_tile_ver = torch.cat([x1,x2], dim=-2)
    print(input_batch.shape, input_tile_hor.shape, input_tile_ver.shape)

    output_batch = net(input_batch)
    output_tile_hor = net(input_tile_hor)
    output_tile_ver = net(input_tile_ver)
    print(output_batch.shape, output_tile_hor.shape, output_tile_ver.shape)

    x1_batch = output_batch[0].unsqueeze(0)
    x2_batch = output_batch[1].unsqueeze(0)

    x1_tile_hor = output_tile_hor[:, :, :, :16]
    x2_tile_hor = output_tile_hor[:, :, :, 16:]

    x1_tile_ver = output_tile_ver[:, :, :16]
    x2_tile_ver = output_tile_ver[:, :, 16:]

    # x1_1 = torch.cat([output_1[0].unsqueeze(0), output_1[1].unsqueeze(0)], dim=-1)

    print(x1_batch.shape, x2_batch.shape, x1_tile_hor.shape, x2_tile_hor.shape, x1_tile_ver.shape, x2_tile_ver.shape)

    diff1 = torch.abs(x1_batch - x1_tile_hor)
    diff2 = torch.abs(x1_batch - x1_tile_ver)
    diff3 = torch.abs(x1_tile_hor - x1_tile_ver)

    print(diff1.sum(), diff2.sum(), diff3.sum())
