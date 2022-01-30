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
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, bias=bias)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bias=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, bias=bias)
        self.down1 = Down(64, 128, bias)
        self.down2 = Down(128, 256, bias)
        self.down3 = Down(256, 512, bias)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, bias)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    net = UNet(1,2, bilinear=False)
    net.eval()
    x1 = torch.zeros((1,1,16,16))
    print(net(x1).shape)
    x2 = torch.zeros((1,1,16,16))

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
