import torch
import torch.nn as nn


class Attention_Gate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_Gate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class RMSA(nn.Module):
    def __init__(self, channels):
        super(RMSA, self).__init__()

        self.channelattention = ChannelAttention(channels=channels)
        self.saptialattention = SpatialAttention(kernel_size=3)

    def forward(self, x):
        inp = x
        x = self.channelattention(x)
        x = self.saptialattention(x)
        return x + inp


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.fc2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        origion = x
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        # final_out = x + x * out
        final_out = origion * out.expand_as(origion)
        return final_out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=2, dilation=2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, kernel_size, padding=3, dilation=3, bias=False)
        self.conv4 = nn.Conv2d(2, 1, kernel_size, padding=4, dilation=4, bias=False)
        self.conv = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        origion = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.conv(out)
        coff = self.sigmoid(out)
        final = origion * coff.expand_as(origion)
        return final
