import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, channel):
        super(FPN, self).__init__()
        self.toplayer =nn.Conv2d(4 * channel, channel, kernel_size=1, stride=1, padding=0)
        self.smooth = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=0)
        self.midlayer = nn.Conv2d(2 * channel, channel, kernel_size=1, stride=1, padding=0)
        self.lowlayer = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def up_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, x0, x1, x2):
        p0 = self.toplayer(x2)
        p1 = self.up_add(p0, self.midlayer(x1))
        p2 = self.up_add(p1, self.lowlayer(x0))
        # p2 = self.smooth(p2)

        return p2