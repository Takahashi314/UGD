# -*- coding: utf-8 -*-
"""
@author lizheng
@date 2022年12月02日 01:21:26
@packageName 
@className Multi_classification_task
@version 1.0.0
@describe Done
"""

# import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Multi_Classification_Type(nn.Module):
    def __init__(self, feature_size, distortion_type_len, batch_num):
        """
        Subtask1 失真类别检测
        :param feature_size: 输入特征向量尺寸
        :param distortion_type_len: 失真类别数量
        """
        super(Multi_Classification_Type, self).__init__()
        self.feature_size = feature_size
        self.distortion_type_len = distortion_type_len
        self.batch_num = batch_num

        # dgcnn
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, self.distortion_type_len)
        self.bn1 = nn.BatchNorm1d(512)

        # # cc network
        # self.fc1 = nn.Linear(self.feature_size, 64)
        # self.fc2 = nn.Linear(64, self.distortion_type_len)
        # self.bn1 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        x = F.softmax(x, dim=1)
        return x


class Multi_Classification_Level(nn.Module):
    def __init__(self, feature_size, distortion_type_len, distortion_level_len, patch_num):
        """
        Subtask2 失真等级检测
        :param feature_size: 输入特性尺寸
        :param distortion_type_len: 失真类型数量
        :param distortion_level_len: 失真等级数量
        :param patch_num: patch_num
        """
        super(Multi_Classification_Level, self).__init__()
        self.feature_size = feature_size
        self.distortion_type_len = distortion_type_len
        self.distortion_level_len = distortion_level_len
        self.patch_num = patch_num

        self.fc1 = nn.Linear(self.feature_size, self.feature_size)
        self.fc2 = nn.Linear(self.feature_size, self.distortion_level_len * self.distortion_type_len)

        self.bn1 = nn.BatchNorm1d(self.feature_size)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        x = x.view(-1, self.distortion_level_len, self.distortion_type_len)
        return x


class Multi_Classification_Level_No_Weighted(nn.Module):
    def __init__(self, feature_size, distortion_level_len):
        """
        Subtask2 失真等级检测
        :param feature_size: 输入特性尺寸
        :param distortion_level_len: 失真等级数量
        """
        super(Multi_Classification_Level_No_Weighted, self).__init__()
        self.feature_size = feature_size
        self.distortion_level_len = distortion_level_len

        # cc network
        self.fc1 = nn.Linear(self.feature_size, 64)
        self.fc2 = nn.Linear(64, self.distortion_level_len)

        self.bn1 = nn.BatchNorm1d(64)

        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        # x = self.dropout(self.fc2(x))
        x = self.fc2(x)
        # x = x.view(-1, self.distortion_level_len, self.distortion_type_len)
        x = F.softmax(x, dim=1)
        return x


class Classification_No_weighted(nn.Module):
    def __init__(self, feature_size, distortion_type_len, distortion_level_len, is_pre_training, patch_num,
                 model='all'):
        """
                Subtask 子任务类
                :param feature_size: 输入特性尺寸
                :param distortion_type_len: 失真类别数量
                :param distortion_level_len: 失真等级数量
                :param is_pre_training: 是否预训练
                """
        super(Classification_No_weighted, self).__init__()
        self.feature_size = feature_size
        self.distortion_type_len = distortion_type_len
        self.distortion_level_len = distortion_level_len
        self.patch_num = patch_num

        self.type_model = Multi_Classification_Type(self.feature_size, self.distortion_type_len, self.patch_num)
        self.level_model = Multi_Classification_Level_No_Weighted(self.feature_size, self.distortion_level_len)

        self.is_pre_training = is_pre_training
        self.model = model

    def set_pre_training_off(self):
        """
        关闭预训练模式
        :return:
        """
        self.is_pre_training = False
        print(self.is_pre_training)

    def forward(self, x):
        """
        :param x: input
        :return: pred_type 经过softmax的预测类别（相当于失真权重）  pred_level 原始数据失真等级
        size: distortion_level_len * distortion_type_len
        """
        pred_type, pred_level = None, None
        if self.is_pre_training:
            pred_type = self.type_model(x)  # softmax(feature)
        else:
            if self.model == 'all':
                pred_type = self.type_model(x)
                pred_level = self.level_model(x)
            if self.model == 'type':
                pred_type = self.type_model(x)
            if self.model == 'level':
                pred_level = self.level_model(x)

        return pred_type, pred_level

    def get_loss(self, weight_lambda, pred_type, pred_level, label_type, label_level):
        """
         Get loss
        :param weight_lambda: 超参数 损失权重
        :param pred_type:
        :param pred_level:
        :param label_type: label_type true
        :param label_level: label_level true
        :return:
        """
        loss1, loss2 = 0, 0
        if self.is_pre_training:
            loss1 = F.nll_loss(torch.log(pred_type + 1e-45), label_type)
        else:
            # todo find nan value
            if torch.isinf(torch.log(pred_type)).sum() or torch.isinf(torch.log(pred_type + 1e-45)).sum():
                print("find nan value!!!!______")
                print('pred_type\n', pred_type)
                print('pred_type+ 1e-50\n', pred_type + 1e-45)
                print('torch.log(pred_type)\n', torch.log(pred_type))
                print('torch.log(pred_type+1e-50)\n', torch.log(pred_type + 1e-45))
                print('label_type\n', label_type)
                print('loss1:\n', F.nll_loss(torch.log(pred_type), label_type))
            if self.model == 'all':
                loss1 = F.nll_loss(torch.log(pred_type + 1e-45), label_type)
                loss2 = F.nll_loss(torch.log(pred_level + 1e-45), label_level)
            if self.model == 'type':
                loss1 = F.nll_loss(torch.log(pred_type + 1e-45), label_type)
            if self.model == 'level':
                loss2 = F.nll_loss(torch.log(pred_level + 1e-45), label_level)
                pred_level = F.softmax(pred_level, dim=0)

        # TODO 这个不太确定怎么设计   我抄的论文里的
        loss = loss1 + loss2
        return loss, pred_level


class Classification(nn.Module):
    def __init__(self, feature_size, distortion_type_len, distortion_level_len, is_pre_training, patch_num):
        """
        Subtask 子任务类
        :param feature_size: 输入特性尺寸
        :param distortion_type_len: 失真类别数量
        :param distortion_level_len: 失真等级数量
        :param is_pre_training: 是否预训练
        """
        super(Classification, self).__init__()
        self.feature_size = feature_size
        self.distortion_type_len = distortion_type_len
        self.distortion_level_len = distortion_level_len
        self.patch_num = patch_num

        self.type_model = Multi_Classification_Type(self.feature_size, self.distortion_type_len, self.patch_num)
        self.level_model = Multi_Classification_Level(self.feature_size, self.distortion_type_len,
                                                      self.distortion_level_len, self.patch_num)

        self.is_pre_training = is_pre_training

    def set_pre_training_off(self):
        """
        关闭预训练模式
        :return:
        """
        self.is_pre_training = False
        print(self.is_pre_training)

    def forward(self, x):
        """
        :param x: input
        :return: pred_type 经过softmax的预测类别（相当于失真权重）  pred_level 原始数据失真等级
        size: distortion_level_len * distortion_type_len
        """
        pred_type, pred_level = None, None
        if self.is_pre_training:
            pred_type = self.type_model(x)  # softmax(feature)
        else:
            pred_type = self.type_model(x)
            pred_level = self.level_model(x)

        return pred_type, pred_level

    def get_loss(self, weight_lambda, pred_type, pred_level, label_type, label_level):
        """
         Get loss
        :param weight_lambda: 超参数 损失权重
        :param pred_type:
        :param pred_level:
        :param label_type: label_type true
        :param label_level: label_level true
        :return:
        """
        loss1, loss2 = 0, 0
        if self.is_pre_training:
            loss1 = F.nll_loss(torch.log(pred_type + 1e-45), label_type)
        else:
            # todo find nan value
            if torch.isinf(torch.log(pred_type)).sum() or torch.isinf(torch.log(pred_type + 1e-45)).sum():
                print("find nan value!!!!______")
                print('pred_type\n', pred_type)
                print('pred_type+ 1e-50\n', pred_type + 1e-45)
                print('torch.log(pred_type)\n', torch.log(pred_type))
                print('torch.log(pred_type+1e-50)\n', torch.log(pred_type + 1e-45))
                print('label_type\n', label_type)
                print('loss1:\n', F.nll_loss(torch.log(pred_type), label_type))
            loss1 = F.nll_loss(torch.log(pred_type + 1e-45), label_type)

            weighted_level = torch.zeros_like(pred_level)
            for i in range(pred_level.shape[0]):  # 对每个batch进行运算  计算类别分类权重*等级权重
                weight_mini_type = torch.diag(pred_type[i])  # 利用对角矩阵右乘 相当于对列做变换
                weighted_level[i] = torch.mm(pred_level[i], weight_mini_type)  # 每一个小batch右乘权重  得到加权的level分类
            pred_level = torch.sum(weighted_level, dim=2)  # 对type求和    size:patch_num * level_len
            loss2 = F.cross_entropy(pred_level, label_level)
            pred_level = F.softmax(pred_level, dim=0)

        # TODO 这个不太确定怎么设计   我抄的论文里的
        loss = loss1 + weight_lambda * loss2
        return loss, pred_level


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    subtask = Classification(64, 7, 11, True, 64)
    torch.autograd.set_detect_anomaly(True)

    test_x = torch.rand((64, 64), dtype=torch.float32)
    test_true_type = torch.ones(64, dtype=torch.long)

    subtask.set_pre_training_off()

    pre_type, pre_level = subtask(test_x)
    print('pre_type.shape:', pre_type.shape, 'pre_level.shape', pre_level.shape)
    loss, _ = subtask.get_loss(0.3, pre_type, pre_level, test_true_type, test_true_type)

    loss.backward()
