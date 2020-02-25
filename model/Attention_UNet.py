import torch
import torch.nn as nn

from module.convolution import conv_block, UNetUp_CT, UNetDsv
from module.attention import UNetGridGatingSignal, GridAttentionBlock2D, init_weights

class Attn_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, nonlocal_mode='concatenation', attention_dsample=2):
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

        self.attn_block1 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                               nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attn_block2 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                               nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attn_block3 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                               nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.up1 = UNetUp_CT(filters[4], filters[3])
        self.up2 = UNetUp_CT(filters[3], filters[2])
        self.up3 = UNetUp_CT(filters[2], filters[1])
        self.up4 = UNetUp_CT(filters[1], filters[0])

        #deep supervision
        self.dsv1 = UNetDsv(in_channels=filters[3], out_channels=self.out_channels, scale_factor=8)
        self.dsv2 = UNetDsv(in_channels=filters[2], out_channels=self.out_channels, scale_factor=4)
        self.dsv3 = UNetDsv(in_channels=filters[1], out_channels=self.out_channels, scale_factor=2)
        self.dsv4 = nn.Conv2d(in_channels=filters[0], out_channels=out_channels, kernel_size=1)

        self.final = nn.Conv2d(self.out_channels*4, out_channels, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        gating = self.gating(center)

        g_conv1, attn1 = self.attn_block1(conv4, gating)
        up1 = self.up1(g_conv1, center)
        g_conv2, attn2 = self.attn_block2(conv3, up1)
        up2 = self.up2(g_conv2, up1)
        g_conv3, attn3 = self.attn_block3(conv2, up2)
        up3 = self.up3(g_conv3, up2)
        up4 = self.up4(conv1, up3)

        dsv1 = self.dsv1(up1)
        dsv2 = self.dsv2(up2)
        dsv3 = self.dsv3(up3)
        dsv4 = self.dsv4(up4)
        final = self.final(torch.cat([dsv4, dsv3, dsv2, dsv1], dim=1))

        return final




class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block = GridAttentionBlock2D(
            in_channels=in_size,
            gating_channels=gate_size,
            inter_channels=inter_size,
            mode=nonlocal_mode,
            sub_sample_factor=sub_sample_factor
        )
        self.combine_gates = nn.Sequential(
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True)
        )
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock2D') != -1: continue
            init_weights(m, 'kaiming')

    def forward(self, input, gating_signal):
        gate, attention = self.gate_block(input, gating_signal)
        return self.combine_gates(gate), attention