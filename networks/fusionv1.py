from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from fu_layers import *
from torch.nn import BatchNorm2d as bn

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.conv1 = ConvBlock(9, 64)
        self.bn1 = bn(64)
        self.conv2 = ConvBlock(64, 128)
        self.bn2 = bn(128)
        self.conv3 = ConvBlock(128, 64)
        self.bn3 = bn(64)
        self.conv4 = ConvBlock(64, 3)
        self.bn4 = bn(3)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_f, s_f, t_f, d):
        all_f = torch.cat((f_f, s_f, t_f, d), 1)
        feature1 = self.relu(self.bn1(self.conv1(all_f)))
        feature2 = self.relu(self.bn2(self.conv2(feature1)))
        feature3 = self.relu(self.bn3(self.conv3(feature2)))
        feature4 = self.relu(self.bn4(self.conv4(feature3)))

        target_frame = self.sigmoid(feature4)
        return target_frame



        

        