import torch
import torch.nn as nn
import torch.nn.functional as F
from module.attention import init_weights

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        result = self.conv(x)
        return result

class up_sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_sample, self).__init__()

        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        result = self.up_sample(x)
        return result

class UNetUp_CT(nn.Module):
    '''code from https://github.com/ozan-oktay/Attention-Gated-Networks/'''
    def __init__(self, in_channels, out_channels):
        super(UNetUp_CT, self).__init__()
        self.conv = conv_block(in_channels+out_channels, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')

        for m in self.children():
            if m.__class__.__name__.find('conv_block') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input1, input2):
        output2 = self.up(input2)
        offset = output2.size()[2] - input1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        output1 = F.pad(input1, padding)
        return self.conv(torch.cat([output1, output2], 1))

class UNetDsv(nn.Module):
    '''code from https://github.com/ozan-oktay/Attention-Gated-Networks/'''
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UNetDsv, self).__init__()
        self.dsv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='trilinear')
        )

    def forward(self, inputs):
        output = self.dsv(inputs)
        return output

class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                out = self.conv(x)
            out = self.conv(x + out)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out