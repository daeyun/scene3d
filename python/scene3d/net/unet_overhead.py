import torch
import numpy as np
import torch.nn as nn


def conv_0(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=8, dilation=4),
        nn.BatchNorm2d(out_ch, momentum=0.002),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=8, dilation=4),
        nn.BatchNorm2d(out_ch, momentum=0.002),
        nn.ReLU(inplace=True),
    )


def conv_1(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(out_ch, momentum=0.002),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch, momentum=0.002),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch, momentum=0.002),
        nn.ReLU(inplace=True),
    )


def up(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_ch, momentum=0.005),
        nn.ReLU(inplace=True),
    )


class Unet0(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.pad = torch.nn.ConstantPad2d(2, 0.0)
        self.unpad = torch.nn.ConstantPad2d(-2, 0.0)

        ch = [64, 64, 64, 256, 512]

        self.enc1 = conv_0(64, ch[0])
        self.enc2 = conv_0(ch[0], ch[1])
        self.enc3 = conv_1(ch[1], ch[2])
        self.enc4 = conv_1(ch[2], ch[3])
        self.enc5 = conv_1(ch[3], ch[4])

        self.dec1 = conv_1(ch[4] + ch[3], ch[3])
        self.dec2 = conv_1(ch[3] + ch[2], ch[2])
        self.dec3 = conv_0(ch[2] + ch[1], ch[1])
        self.dec4 = conv_0(ch[1] + ch[0], ch[0])

        self.dec5 = nn.Conv2d(ch[0], out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        padded_x = self.pad(x)

        x1 = self.enc1(padded_x)  # (304, 304)
        x2 = self.enc2(self.pool(x1))  # (152, 152)
        x3 = self.enc3(self.pool(x2))  # (76, 76)
        x4 = self.enc4(self.pool(x3))  # (38, 38)
        out = self.enc5(self.pool(x4))  # (19, 19)

        out = self.dec1(torch.cat([x4, self.unpool(out)], dim=1))  # (38, 38)
        out = self.dec2(torch.cat([x3, self.unpool(out)], dim=1))  # (76, 76)
        out = self.dec3(torch.cat([x2, self.unpool(out)], dim=1))  # (152, 152)
        out = self.dec4(torch.cat([x1, self.unpool(out)], dim=1))  # (304, 304)
        out = self.dec5(out)  # (304, 304)
        out = self.unpad(out)  # (300, 300)

        return out