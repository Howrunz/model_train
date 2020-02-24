import torch
import torch.nn as nn

from module.convolution import conv_block, up_sample
from module.attention import UNetGridGatingSignal,

class Attn_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Attn_UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        features = 32
        filters = [features, features*2, features*4, features*8, features*16]

        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = conv_block(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = conv_block(filters[3], filters[4])
        self.gating = UNetGridGatingSignal(filters[4], filters[4])

        self.attn_block1 =

class MultiAttentionBlock(nn.Module):
    def __init__(self):