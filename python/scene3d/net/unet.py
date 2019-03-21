import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional
from third_party.inplace_abn.modules import InPlaceABN, InPlaceABNSync


def conv_0(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=8, dilation=4, bias=False),
        nn.BatchNorm2d(out_ch, momentum=0.005),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=8, dilation=4, bias=False),
        nn.BatchNorm2d(out_ch, momentum=0.005),
        nn.ReLU(inplace=True),
    )


def conv_0_ip(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=8, dilation=4, bias=False),
        InPlaceABNSync(out_ch, momentum=0.005, activation="leaky_relu", slope=0.01),
        nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=8, dilation=4, bias=False),
        InPlaceABNSync(out_ch, momentum=0.005, activation="leaky_relu", slope=0.01),
    )


def conv_1(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2, bias=False),
        nn.BatchNorm2d(out_ch, momentum=0.005),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch, momentum=0.005),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch, momentum=0.005),
        nn.ReLU(inplace=True),
    )


def conv_1_ip(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2, bias=False),
        InPlaceABNSync(out_ch, momentum=0.005, activation="leaky_relu", slope=0.01),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        InPlaceABNSync(out_ch, momentum=0.005, activation="leaky_relu", slope=0.01),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        InPlaceABNSync(out_ch, momentum=0.005, activation="leaky_relu", slope=0.01),
    )


def up(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_ch, momentum=0.005),
        nn.ReLU(inplace=True),
    )


class Unet0(nn.Module):
    def __init__(self, out_channels=1, ch=(64, 64, 64, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.enc1 = conv_0(3, ch[0])
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
        x1 = self.enc1(x)  # (240, 320)
        x2 = self.enc2(self.pool(x1))  # (120, 160)
        x3 = self.enc3(self.pool(x2))  # (60, 80)
        x4 = self.enc4(self.pool(x3))  # (30, 40)
        out = self.enc5(self.pool(x4))  # (15, 20)

        out = self.dec1(torch.cat([x4, self.unpool(out)], dim=1))  # (30, 40)
        out = self.dec2(torch.cat([x3, self.unpool(out)], dim=1))  # (60, 80)
        out = self.dec3(torch.cat([x2, self.unpool(out)], dim=1))  # (120, 160)
        out = self.dec4(torch.cat([x1, self.unpool(out)], dim=1))  # (240, 320)
        out = self.dec5(out)  # (240, 320)
        return out


class Unet1(nn.Module):
    def __init__(self, out_channels=1, in_channels=3, ch=(64, 64, 64, 384, 768)):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)

        self.enc1 = conv_0_ip(in_channels, ch[0])
        self.enc2 = conv_0_ip(ch[0], ch[1])
        self.enc3 = conv_1_ip(ch[1], ch[2])
        self.enc4 = conv_1_ip(ch[2], ch[3])
        self.enc5 = conv_1_ip(ch[3], ch[4])

        self.dec1 = conv_1_ip(ch[4] + ch[3], ch[3])
        self.dec2 = conv_1_ip(ch[3] + ch[2], ch[2])
        self.dec3 = conv_0_ip(ch[2] + ch[1], ch[1])
        self.dec4 = conv_0_ip(ch[1] + ch[0], ch[0])

        self.dec5 = nn.Conv2d(ch[0], out_channels, kernel_size=3, padding=1, bias=False)

    def unpool(self, value):
        # TODO
        return nn.functional.interpolate(value, scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = self.enc1(x)  # (240, 320)
        x2 = self.enc2(self.pool(x1))  # (120, 160)
        x3 = self.enc3(self.pool(x2))  # (60, 80)
        x4 = self.enc4(self.pool(x3))  # (30, 40)
        out = self.enc5(self.pool(x4))  # (15, 20)

        out = self.dec1(torch.cat([x4, self.unpool(out)], dim=1))  # (30, 40)
        out = self.dec2(torch.cat([x3, self.unpool(out)], dim=1))  # (60, 80)
        out = self.dec3(torch.cat([x2, self.unpool(out)], dim=1))  # (120, 160)
        out = self.dec4(torch.cat([x1, self.unpool(out)], dim=1))  # (240, 320)
        out = self.dec5(out)  # (240, 320)
        return out


class Unet2(nn.Module):
    """
    Branched unet.
    """

    def __init__(self, out_channels=1, ch=(48, 64, 64, 384, 768), ch_branch=32, in_channels=3):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)

        self.enc1 = conv_0_ip(in_channels, ch[0])
        self.enc2 = conv_0_ip(ch[0], ch[1])
        self.enc3 = conv_1_ip(ch[1], ch[2])
        self.enc4 = conv_1_ip(ch[2], ch[3])
        self.enc5 = conv_1_ip(ch[3], ch[4])

        self.dec1 = conv_1_ip(ch[4] + ch[3], ch[3])
        self.dec2 = conv_1_ip(ch[3] + ch[2], ch[2])
        self.dec3 = conv_0_ip(ch[2] + ch[1], ch[1])
        self.dec4 = conv_0_ip(ch[1] + ch[0], ch[0])

        self.dec5_branched = nn.ModuleList()
        for i in range(out_channels):
            self.dec5_branched.append(conv_1_ip(ch[0], ch_branch))

        self.dec6_branched = nn.ModuleList()
        for i in range(out_channels):
            self.dec6_branched.append(nn.Conv2d(ch_branch, 1, kernel_size=3, padding=1, bias=False))

    def unpool(self, value):
        return nn.functional.interpolate(value, scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = self.enc1(x)  # (240, 320)
        x2 = self.enc2(self.pool(x1))  # (120, 160)
        x3 = self.enc3(self.pool(x2))  # (60, 80)
        x4 = self.enc4(self.pool(x3))  # (30, 40)
        out = self.enc5(self.pool(x4))  # (15, 20)

        out = self.dec1(torch.cat([x4, self.unpool(out)], dim=1))  # (30, 40)
        out = self.dec2(torch.cat([x3, self.unpool(out)], dim=1))  # (60, 80)
        out = self.dec3(torch.cat([x2, self.unpool(out)], dim=1))  # (120, 160)
        out = self.dec4(torch.cat([x1, self.unpool(out)], dim=1))  # (240, 320)

        out_branched = []
        for i in range(len(self.dec5_branched)):
            out_i = out
            out_i = self.dec5_branched[i](out_i)  # (B, 32, 240, 320)
            out_i = self.dec6_branched[i](out_i)  # (B, 1, 240, 320)
            out_branched.append(out_i)

        out = torch.cat(out_branched, dim=1)  # (B, C, 240, 320)

        return out


class Unet2Regression(nn.Module):
    def __init__(self, out_features=3):
        super().__init__()
        self.out_features = out_features

        channels = [64, 128, 256, 512, 1024, 2048]

        self.enc = nn.Sequential(
            nn.Conv2d(48, channels[0], kernel_size=3, padding=2, dilation=2),
            InPlaceABN(channels[0], momentum=0.01, activation="relu"),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=2, dilation=2),
            InPlaceABN(channels[1], momentum=0.01, activation="relu"),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=2, dilation=2),
            InPlaceABN(channels[2], momentum=0.01, activation="relu"),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=2, dilation=2),
            InPlaceABN(channels[3], momentum=0.01, activation="relu"),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1, dilation=1),
            InPlaceABN(channels[4], momentum=0.01, activation="relu"),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
            nn.Conv2d(channels[4], channels[5], kernel_size=3, padding=0, dilation=1),
            nn.AvgPool2d(kernel_size=(5, 8)),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=768 * 15 * 20, out_features=1024),
            InPlaceABN(1024, momentum=0.01, activation="relu"),
        )

        self.linear_final = nn.Linear(in_features=2048 + 1024, out_features=self.out_features)

    def forward(self, x):
        assert isinstance(x, (tuple, list))
        enc_out = self.enc(x[0])
        linear_out = self.linear(x[1].view(x[1].shape[0], -1))

        assert enc_out.shape[2] == 1
        assert enc_out.shape[3] == 1

        out = torch.cat([enc_out.view(-1, enc_out.shape[1]), linear_out], dim=1)
        out = self.linear_final(out)

        return out


def get_feature_map_output(model: Unet0, x):
    assert str(model.__class__) == str(Unet0)

    x1 = model.enc1(x)  # (240, 320)
    x2 = model.enc2(model.pool(x1))  # (120, 160)
    x3 = model.enc3(model.pool(x2))  # (60, 80)
    x4 = model.enc4(model.pool(x3))  # (30, 40)
    out = model.enc5(model.pool(x4))  # (15, 20)

    out = model.dec1(torch.cat([x4, model.unpool(out)], dim=1))  # (30, 40)
    out = model.dec2(torch.cat([x3, model.unpool(out)], dim=1))  # (60, 80)
    out = model.dec3(torch.cat([x2, model.unpool(out)], dim=1))  # (120, 160)

    # (240, 320)
    out = model.dec4[:3](torch.cat([x1, model.unpool(out)], dim=1))

    return out


def get_feature_map_output_v1(model: Unet1, x, return_final_output=False):
    # assert str(model.__class__) == str(Unet1)

    x1 = model.enc1(x)  # (240, 320)
    x2 = model.enc2(model.pool(x1))  # (120, 160)
    x3 = model.enc3(model.pool(x2))  # (60, 80)
    x4 = model.enc4(model.pool(x3))  # (30, 40)
    out = model.enc5(model.pool(x4))  # (15, 20)

    out = model.dec1(torch.cat([x4, model.unpool(out)], dim=1))  # (30, 40)
    out = model.dec2(torch.cat([x3, model.unpool(out)], dim=1))  # (60, 80)
    out = model.dec3(torch.cat([x2, model.unpool(out)], dim=1))  # (120, 160)

    # (240, 320)
    out = model.dec4[:2](torch.cat([x1, model.unpool(out)], dim=1))

    feat_out = out

    if not return_final_output:
        return feat_out

    out = model.dec4[2:](out)
    out = model.dec5(out)  # (240, 320)
    final_out = out

    return feat_out, final_out


def get_feature_map_output_v2(model: Unet2, x, return_encoding=True, return_final_output=False):
    # assert str(model.__class__) == str(Unet2)

    x1 = model.enc1(x)  # (240, 320)
    x2 = model.enc2(model.pool(x1))  # (120, 160)
    x3 = model.enc3(model.pool(x2))  # (60, 80)
    x4 = model.enc4(model.pool(x3))  # (30, 40)
    out = model.enc5(model.pool(x4))  # (15, 20)

    if return_encoding:
        encoding = out

    out = model.dec1(torch.cat([x4, model.unpool(out)], dim=1))  # (30, 40)
    out = model.dec2(torch.cat([x3, model.unpool(out)], dim=1))  # (60, 80)
    out = model.dec3(torch.cat([x2, model.unpool(out)], dim=1))  # (120, 160)

    # (240, 320)
    out = model.dec4(torch.cat([x1, model.unpool(out)], dim=1))

    if not return_final_output:
        if return_encoding:
            return out, encoding
        else:
            return out

    out_branched = []
    for i in range(len(model.dec5_branched)):
        out_i = out
        out_i = model.dec5_branched[i](out_i)  # (B, 32, 240, 320)
        out_i = model.dec6_branched[i](out_i)  # (B, 1, 240, 320)
        out_branched.append(out_i)

    final_out = torch.cat(out_branched, dim=1)  # (B, C, 240, 320)
    del out_branched

    if return_encoding:
        return out, encoding, final_out
    else:
        return out, final_out
