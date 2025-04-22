# -*- coding: utf-8 -*-
"""
@author lizheng
@date 2023年01月27日 20:02:55
@packageName 
@className Regression
@version 1.0.0
@describe TODO
"""

import torch
from torch import nn
import torch.nn.functional as F


class Regression(nn.Module):
    def __init__(self, feature_size, batch_num):
        super(Regression, self).__init__()
        self.feature_size = feature_size
        self.batch_num = batch_num

        # cc network
        self.fc1 = nn.Linear(self.feature_size, int(self.feature_size / 2))
        self.fc2 = nn.Linear(int(self.feature_size / 2), int(self.feature_size / 4))
        self.fc3 = nn.Linear(int(self.feature_size / 4), int(self.feature_size / 8))

        self.pred = nn.Linear(int(self.feature_size / 8), 1)

        self.bn1 = nn.BatchNorm1d(int(self.feature_size / 2))
        self.bn2 = nn.BatchNorm1d(int(self.feature_size / 4))
        self.bn3 = nn.BatchNorm1d(int(self.feature_size / 8))

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.pred(x)

        return x


if __name__ == '__main__':
    regression = Regression(128, 4)
    test_feature = torch.randn((4, 128))
    print('text_feature.shape =', test_feature.shape)

    pred = regression(test_feature)
    print('pred.shape =', pred.shape)


