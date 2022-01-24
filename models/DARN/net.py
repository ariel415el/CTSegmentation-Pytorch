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

    def __init__(self, in_channels, out_channels, trilinear=False, bias=False):
        super().__init__()
        if trilinear:
            self.layers = [nn.MaxPool3d(kernel_size=2, stride=2, padding=0)]
        else:
            self.layers = [nn.Conv3d(in_channels, in_channels, kernel_size=2, padding=1, bias=bias, stride=2)]
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


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=True, bias=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv(n_channels, 64, bias)
        self.down1 = Down(64, 128, bias)
        self.down2 = Down(128, 256, bias)
        self.down3 = Down(256, 512, bias)
        factor = 2 if trilinear else 1
        self.down4 = Down(512, 1024 // factor, bias)
        self.up1 = Up(1024, 512 // factor, trilinear)
        self.up2 = Up(512, 256 // factor, trilinear)
        self.up3 = Up(256, 128 // factor, trilinear)
        self.up4 = Up(128, 64, trilinear)

    def forward(self, x):
        x1 = self.inc(x.unsqueeze(1))
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
    net = UNet(1,2)
    net.eval()
    x1 = torch.zeros((2,1,16,32,32))
    x2 = torch.zeros((2,1,16,32,32))

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