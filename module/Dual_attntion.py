import torch
import torch.nn as nn

### code from https://github.com/junfu1115/DANet/
class PAM(nn.Module):
    '''Position Attention Module'''
    def __init__(self, in_channel):
        super(PAM, self).__init__()
        self.in_channel = in_channel
        self.query_conv = nn.Conv2d(
            in_channels=self.in_channel,
            out_channels=self.in_channel//8,
            kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=self.in_channel,
            out_channels=self.in_channel // 8,
            kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=self.in_channel,
            out_channels=self.in_channel,
            kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.Softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out

class CAM(nn.Module):
    '''Channel Attention Module'''
    def __init__(self, in_channel):
        super(CAM, self).__init__()
        self.in_channel = in_channel
        self.gamma = nn.Parameter(torch.zeros(1))
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = x.view(batch_size, C, -1)
        proj_key = x.view(batch_size, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.Softmax(energy_new)
        proj_value = x.view(batch_size, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out

