import torch
import torch.nn as nn
import torch.nn.functional as F
from module.convolution import *

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=32):
        super(UNet, self).__init__()

        self.conv1 = conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = conv_block(features *4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(features * 8, features * 16)

        self.up1 = up_sample(features * 16, features * 8)
        self.up_conv1 = conv_block(features * 16, features * 8)
        self.up2 = up_sample(features * 8, features * 4)
        self.up_conv2 = conv_block(features * 8, features * 4)
        self.up3 = up_sample(features * 4, features * 2)
        self.up_conv3 = conv_block(features * 4, features * 2)
        self.up4 = up_sample(features * 2, features)
        self.up_conv4 = conv_block(features * 2, features)
        self.conv5 = nn.Conv2d(features, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        bottleneck = self.bottleneck(pool4)

        up1 = self.up1(bottleneck)
        concat1 = torch.cat((conv4, up1), dim=1)
        up_conv1 = self.up_conv1(concat1)
        up2 = self.up2(up_conv1)
        concat2 = torch.cat((conv3, up2), dim=1)
        up_conv2 = self.up_conv2(concat2)
        up3 = self.up3(up_conv2)
        concat3 = torch.cat((conv2, up3), dim=1)
        up_conv3 = self.up_conv3(concat3)
        up4 = self.up4(up_conv3)
        concat4 = torch.cat((conv1, up4), dim=1)
        up_conv4 = self.up_conv4(concat4)
        out = self.conv5(up_conv4)

        return torch.sigmoid(out)