from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1), nn.ReLU(True), nn.Conv2d(c // r, c, 1), nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, attn=False):
        super().__init__()
        self.conv1 = ConvBNReLU(in_c, out_c)
        self.conv2 = ConvBNReLU(out_c, out_c)
        self.attn = SEBlock(out_c) if attn else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.attn(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, features=(64, 128, 256, 512), attn=True):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Down path
        ch = in_channels
        for f in features:
            self.downs.append(UNetBlock(ch, f, attn=attn))
            ch = f
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2, attn=attn)

        # Up path
        rev = list(reversed(features))
        up_c = features[-1] * 2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(up_c, f, kernel_size=2, stride=2))
            self.ups.append(UNetBlock(up_c, f, attn=attn))
            up_c = f

        self.head = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.head(x)
