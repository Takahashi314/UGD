# -*- coding: utf-8 -*-
"""
@author lizheng
@date 2023年03月29日 22:54:29
@packageName 
@className subtask
@version 1.0.0
@describe TODO
"""

import torch.nn.functional as F
import torch
from torch import nn
import argparse


class Weight_Net(nn.Module):
    def __init__(self, inp_len=2048, oup_len=1):
        super(Weight_Net, self).__init__()
        self.MLPLayers1 = nn.Sequential(
            nn.Linear(in_features=inp_len, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64),
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


class Classification_Type(nn.Module):
    def __init__(self, distortion_type_len, inp_len=2048):
        super(Classification_Type, self).__init__()
        self.MLPShareLayers = nn.Sequential(
            nn.Linear(in_features=inp_len, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.classification_head = nn.Linear(in_features=64, out_features=distortion_type_len)
        self.distribution_head = nn.Sequential(
            nn.Linear(in_features=64, out_features=distortion_type_len - 1),
            nn.Softmax()
        )

    def forward(self, x):
        """
        :param x:   N x C
        :return:    N x 1
        """
        x = self.MLPShareLayers(x)
        cls_out = self.classification_head(x)
        distribution_out = self.distribution_head(x)
        return cls_out, distribution_out


class BasicFCModule(nn.Module):
    def __init__(self, inp_len=2048):
        super(BasicFCModule, self).__init__()
        self.MLPLayers1 = nn.Sequential(
            nn.Linear(in_features=inp_len, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64),
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


class weight_score(nn.Module):
    def __init__(self, batch_size, patch_number, inp_len=2048):
        super(weight_score, self).__init__()
        self.score_mlp = BasicFCModule(inp_len)
        self.patch_number = patch_number
        self.batch_size = batch_size

        # self.feature = my_pointnet_deep()
        # self.score_mlp = BasicFCModule(960)
        # self.weight_mlp = BasicFCModule(960)

    def forward(self, fea_vector, weight):
        """
        :param weight:
        :param fea_vector: B x patch_number x D x patch_size
        :return: B x 1
        """
        score = self.score_mlp(fea_vector)  # (B x patch_number) x 1
        score = score.view(self.batch_size, self.patch_number)
        weight = weight.view(self.batch_size, self.patch_number)  # B x patch_number
        product_val = torch.mul(score, weight)
        product_val_sum = torch.sum(product_val, dim=-1)
        norm_val = torch.sum(weight, dim=-1)
        final_score = torch.div(product_val_sum, norm_val)
        final_score = final_score.view(self.batch_size, -1)  # B x 1
        # final_score = torch.mean(score,dim=-1)
        # final_score = final_score.view(B, -1)
        return final_score


class Pair_Net(nn.Module):
    def __init__(self, batch_size, patch_number, inp_len=2048):
        super(Pair_Net, self).__init__()
        self.score_compute = weight_score(batch_size, patch_number, inp_len=inp_len)

    @staticmethod
    def fusion_layer(x1, x2):
        difference = x1 - x2
        out = torch.div(1, 1 + torch.exp(-difference))
        return out, difference

    def forward(self, x1, x2, weight1, weight2):
        """
        :param weight2:
        :param weight1:
        :param x1:  B x patch_number x D x patch_size
        :param x2:   B x patch_number x D x patch_size
        :return: B x 1
        """
        score1 = self.score_compute(x1, weight1)
        score2 = self.score_compute(x2, weight2)
        # if np.any(np.isnan(score1.cpu().detach().numpy())) or np.any(np.isnan(score1.cpu().detach().numpy())):
        #     print("score is nan")
        x, difference = self.fusion_layer(score1, score2)  # B x 1
        return score1, score2, x, difference


class Regression(nn.Module):
    def __init__(self, fec_vector_len=2048):
        super(Regression, self).__init__()
        self.MLPLayers = nn.Sequential(
            nn.Linear(in_features=fec_vector_len, out_features=fec_vector_len // 2),
            nn.BatchNorm1d(fec_vector_len // 2),
            # nn.ReLU(),
            # nn.Linear(in_features=1024, out_features=512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Linear(in_features=512, out_features=256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Linear(in_features=256, out_features=64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=8),
            # nn.BatchNorm1d(8,),
            # nn.ReLU(),
            nn.Linear(in_features=fec_vector_len // 2, out_features=1),
        )

    def forward(self, x):
        return self.MLPLayers(x)


if __name__ == '__main__':
    t = torch.randn((8, 2048))
    print(t.shape)
    r = Regression()
    rst = r(t)
    print(rst.shape)
