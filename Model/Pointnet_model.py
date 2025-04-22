# -*- coding: utf-8 -*-
"""
@author lizheng
@date 2023年02月03日 22:26:39
@packageName 
@className Pointnet_model
@version 1.0.0
@describe TODO
"""
from torch import nn
import torch.nn.functional as F
from R_Quality_Assessment.SSL_Multitasking.Model.pointnet_utils import PointNetEncoder
import torch


class pointnet_extractor_model(nn.Module):
    def __init__(self, k=11, normal_channel=True):
        super(pointnet_extractor_model, self).__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    feature_extractor = pointnet_extractor_model()
    t = torch.randn((2, 3, 8192))
    print(t.shape)
    feature = feature_extractor(t)
    print(feature.shape)
