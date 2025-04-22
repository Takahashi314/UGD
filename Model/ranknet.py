# -*- coding: utf-8 -*-
"""
@author lizheng
@date 2023年03月02日 21:48:09
@packageName 
@className ranknet
@version 1.0.0
@describe TODO
"""
from torch import nn
import torch
import numpy as np


def refresh_pairwise_correct_list(arr, dist1, dist2, label):
    """
    :param arr: pairwise_correct_list
    :param dist1:
    :param dist2:
    :param label: true label
    :return:
    """
    dist1, dist2, label = \
        dist1.cpu().detach().numpy(), dist2.cpu().detach().numpy(), label.cpu().detach().numpy()  # Turn tensor to numpy
    pred = np.sum(dist1 > dist2, axis=1, keepdims=True)  # Turn score to pred label
    correct_new = (~np.logical_xor(pred, label)).flatten()  # Calculate correct label
    return np.concatenate((arr, correct_new))


class BasicFCModule(nn.Module):
    def __init__(self, inp_len=2048, oup_len=1):
        super(BasicFCModule, self).__init__()
        self.MLPLayers1 = nn.Sequential(
            nn.Linear(in_features=inp_len, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        :param x:   N x C
        :return:    N x 1
        """
        x = self.MLPLayers1(x)
        return x


class Pairwise_Net(nn.Module):
    def __init__(self):
        super(Pairwise_Net, self).__init__()
        self.score_compute = BasicFCModule()
        self.criterion = nn.BCELoss()

    @staticmethod
    def fusion_layer(x1, x2):
        difference = x1 - x2
        out = torch.div(1, 1 + torch.exp(-difference))
        return out

    def forward(self, x1, x2):
        """
        :param x1:  B x patch_number x D x patch_size
        :param x2:   B x patch_number x D x patch_size
        :return: B x 1
        """
        score1 = self.score_compute(x1)
        score2 = self.score_compute(x2)
        # if np.any(np.isnan(score1.cpu().detach().numpy())) or np.any(np.isnan(score1.cpu().detach().numpy())):
        #     print("score is nan")
        x = self.fusion_layer(score1, score2)  # B x 1
        return score1, score2, x

    def get_loss(self, pred_probability, label):
        return self.criterion(pred_probability, label)


class weight_score(nn.Module):
    def __init__(self, feature_model):
        super(weight_score, self).__init__()
        self.feature = feature_model
        self.score_mlp = BasicFCModule()
        self.weight_mlp = BasicFCModule()

    def forward(self, x):
        """
        :param x: B x patch_number x patch_size x D
        :return: B x 1
        """
        B, patch_number, patch_size, D = x.size()

        x = x.view(-1, D, patch_size)  # pointnet网络输入只能接受3维    (B x patch_number) x D x patch_size
        fea_vector = self.feature(x)  # (B x patch_number) x 1024
        score = self.score_mlp(fea_vector)  # (B x patch_number) x 1
        weight = self.weight_mlp(fea_vector)  # (B x patch_number) x 1  线性层只能接受2维
        score = score.view(B, patch_number)
        weight = weight.view(B, patch_number)  # B x patch_number
        product_val = torch.mul(score, weight)
        product_val_sum = torch.sum(product_val, dim=-1)
        norm_val = torch.sum(weight, dim=-1)
        final_score = torch.div(product_val_sum, norm_val)
        final_score = final_score.view(B, -1)  # B x 1
        # final_score = torch.mean(score,dim=-1)
        # final_score = final_score.view(B, -1)
        return final_score


class Pairwise_Net_patch(nn.Module):
    def __init__(self, feature_model):
        super(Pairwise_Net_patch, self).__init__()
        self.score_compute = weight_score(feature_model)

    def FusionLayer(self, x1, x2):
        difference = x1 - x2
        out = torch.div(1, 1 + torch.exp(-difference))
        return out

    def forward(self, x1, x2):
        """
        :param x1:  B x patch_number x D x patch_size
        :param x2:   B x patch_number x D x patch_size
        :return: B x 1
        """
        score1 = self.score_compute(x1)
        score2 = self.score_compute(x2)
        # if np.any(np.isnan(score1.cpu().detach().numpy())) or np.any(np.isnan(score1.cpu().detach().numpy())):
        #     print("score is nan")
        x = self.FusionLayer(score1, score2)  # B x 1
        return score1, score2, x


if __name__ == '__main__':
    test_feature1 = torch.randn((4, 2048))
    test_feature2 = torch.randn((4, 2048))
    test_label = torch.randn(4).view(4, 1)

    pairnet = Pairwise_Net()

    score1, score2, x = pairnet(test_feature1, test_feature2)
    criterion = nn.BCELoss()

    print('score1\n', score1)
    print('score2\n', score2)
    print('x\n', x)

    loss = criterion(x, test_label)
    print(loss)
