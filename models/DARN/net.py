""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bias=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True, bias=False):
        super().__init__()
        if trilinear:
            self.layers = [nn.MaxPool3d(kernel_size=2, stride=2, padding=0)]
        else:
            self.layers = [nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias, stride=2)]
        self.layers.append(DoubleConv(in_channels, out_channels, bias))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SemRef(nn.Module):
    """
        Uses lower and higher level feature maps to create a channel attention for the low level maps
        returns a tensor of shape lower_level_map
    """
    def __init__(self, low_in_channels, high_in_channels, bias=False):
        super().__init__()
        self.conv_IN_LR = nn.Sequential(
            nn.Conv3d(low_in_channels, low_in_channels, kernel_size=3, padding=1, bias=bias),
            nn.InstanceNorm3d(low_in_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_1x1 = nn.Sequential(
            nn.Conv3d(low_in_channels + high_in_channels, low_in_channels, kernel_size=1, padding=0, bias=bias),
            nn.Softmax()
        )

    def forward(self, lower_level_map, higher_level_map):
        all_maps = torch.cat([lower_level_map, higher_level_map], dim=1)
        attention_weights = torch.mean(all_maps, dim=[-2, -1], keepdim=True) # global_average_pooling
        attention_weights = self.conv_1x1(attention_weights)
        new_maps = lower_level_map + self.conv_IN_LR(lower_level_map) * attention_weights
        return new_maps


class SpaRef(nn.Module):
    """
    Uses lower and higher level feature maps to create a spatial attention for the high level maps
        returns a tensor of shape higher_level_map
    """
    def __init__(self, low_in_channels, high_in_channels, bias=False):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv3d(low_in_channels + high_in_channels, high_in_channels, kernel_size=3, padding=1, bias=bias),
            nn.InstanceNorm3d(low_in_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv3d(high_in_channels, 1, kernel_size=3, padding=1, bias=bias),
            nn.Softmax()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv3d(high_in_channels, high_in_channels, kernel_size=3, padding=1, bias=bias),
            nn.InstanceNorm3d(high_in_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, lower_level_map, higher_level_map):
        all_maps = torch.cat([lower_level_map, higher_level_map], dim=1)
        attention_map = self.conv_2(self.conv_1(all_maps))
        weighted_higher_map = higher_level_map + attention_map * higher_level_map
        new_maps = self.conv_3(weighted_higher_map)
        return new_maps


class DARN(nn.Module):
    def __init__(self, n_classes, trilinear=True, bias=False):
        super(DARN, self).__init__()
        self.n_classes = n_classes
        self.trilinear = trilinear
        p = 32
        self.inc = DoubleConv(1, p, bias)
        self.down1 = Down(p, p*2, bias)
        self.down2 = Down(p*2, p*4, bias)
        self.down3 = Down(p*4, p*8, bias)
        factor = 2 if trilinear else 1
        self.down4 = Down(p*8, p*16 // factor, bias)
        self.up1 = Up(p*16, p*8 // factor, trilinear)
        self.up2 = Up(p*8, p*4 // factor, trilinear)
        self.up3 = Up(p*4, p*2 // factor, trilinear)
        self.up4 = Up(p*2, p, trilinear)

        self.upscale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.upscale_4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.upscale_8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
                                             # nn.Conv3d(p*8 // factor, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)

        self.sem_ref0 = SemRef(p, p)
        self.sem_ref1 = SemRef(p, p*2)
        self.sem_ref2 = SemRef(p*2, p*4)

        self.spa_ref1 = SpaRef(p, p)
        self.spa_ref2 = SpaRef(p, p*2)
        self.spa_ref3 = SpaRef(p*2, p*4)

        self.final_out_conv = nn.Conv3d(p + p*2 + p*4, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)

        self.maps1_l3_out = nn.Conv3d(p*4, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)
        self.maps1_l2_out = nn.Conv3d(p*2, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)
        self.maps1_l1_out = nn.Conv3d(p, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)
        self.maps1_l0_out = nn.Conv3d(p, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)

        self.maps2_l3_out = nn.Conv3d(p*4, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)
        self.maps2_l2_out = nn.Conv3d(p*2, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)
        self.maps2_l1_out = nn.Conv3d(p, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)
        self.maps2_l0_out = nn.Conv3d(p, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)

        self.maps3_l3_out = nn.Conv3d(p*4, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)
        self.maps3_l2_out = nn.Conv3d(p*2, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)
        self.maps3_l1_out = nn.Conv3d(p, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)
        self.maps3_l0_out = nn.Conv3d(p, n_classes, kernel_size=1, padding=0, bias=bias, stride=1)

    def forward(self, x):
        x = x.unsqueeze(1)                # (b, 1, s, h, w)
        x_l0 = self.inc(x)                # (b, p, s, h, w)
        x_l1 = self.down1(x_l0)           # (b, p*2, s//2, h//2, w//2)
        x_l2 = self.down2(x_l1)           # (b, p*4, s//4, h//4, w//4)
        x_l3 = self.down3(x_l2)           # (b, p*8, s//8, h//8, w//8)
        y_l4 = self.down4(x_l3)           # (b, p*8, s//16, h//16, w//16)

        # decoder
        y_l3 = self.up1(y_l4, x_l3)       # (b, p*4, s//8, h//8, w//8)
        y_l2 = self.up2(y_l3, x_l2)       # (b, p*2, s//4, h//4, w//4)
        y_l1 = self.up3(y_l2, x_l1)       # (b, p, s//2, h//2, w//2)
        y_l0 = self.up4(y_l1, x_l0)       # (b, p, s, h, w)

        maps1_l3 = self.upscale_8(y_l3)   # (b, p*4, s, h, w)
        maps1_l2 = self.upscale_4(y_l2)   # (b, p*2, s, h, w)
        maps1_l1 = self.upscale_2(y_l1)   # (b, p, s, h, w)
        maps1_l0 = y_l0                   # (b, p, s, h, w)

        maps2_l3 = maps1_l3                           # (b, p*4, s, h, w)
        maps2_l2 = self.sem_ref2(maps1_l2, maps1_l3)  # (b, p*2, s, h, w)
        maps2_l1 = self.sem_ref1(maps1_l1, maps1_l2)  # (b, p, s, h, w)
        maps2_l0 = self.sem_ref0(maps1_l0, maps1_l1)  # (b, p, s, h, w)

        maps3_l3 = self.spa_ref3(maps2_l2, maps2_l3)  # (b, p*4, s, h, w)
        maps3_l2 = self.spa_ref2(maps2_l1, maps2_l2)  # (b, p*2, s, h, w)
        maps3_l1 = self.spa_ref1(maps2_l0, maps2_l1)  # (b, p, s, h, w)
        maps3_l0 = maps2_l0                           # (b, p, s, h, w)

        all_maps = torch.cat([maps3_l3, maps3_l2, maps3_l1], dim=1)
        final_pred = self.final_out_conv(all_maps)

        out_convs = [
            self.maps1_l0_out(maps1_l0), self.maps1_l1_out(maps1_l1), self.maps1_l2_out(maps1_l2), self.maps1_l3_out(maps1_l3),
            self.maps2_l0_out(maps1_l0), self.maps3_l1_out(maps2_l1), self.maps2_l2_out(maps2_l2), self.maps3_l3_out(maps2_l3),
            self.maps3_l0_out(maps1_l0), self.maps2_l1_out(maps3_l1), self.maps3_l2_out(maps3_l2), self.maps2_l3_out(maps3_l3)
        ]

        return final_pred, out_convs


if __name__ == '__main__':
    net = DARN(2)
    nparams = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(nparams)
    net.eval()
    x1 = torch.zeros((3,32,128,128))
    print(net(x1)[0].shape)
