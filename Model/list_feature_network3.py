import os
import sys
import copy
import math

# import chainer
# import chainer.functions as CF
# import chainer.links as CL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# import list_dataloader
# from list_dataloader import PointCloudDataloader
# from torch.utils.data import Dataset, DataLoader
# # from utils_rank import plot_result
# # from utils_rank import NNfuncs


# @data:2022/03/14


def knn(x, k):
    '''

    :param x: [B C N]
    :param k: int
    :return: [B N K]
    '''
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    # print("inner=",inner.shape)
    # print("xx=",xx.shape)
    # print("pairwise_distance=",pairwise_distance.shape)
    # print("idx",idx.shape)
    return idx


def get_graph_feature(x, k=20, idx=None):
    '''

    :param x: [B C N]
    :param k: int
    :param idx:
    :return: [B C*2 N K]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size,num_points,k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    # idx_base = torch.arange(0, batch_size, device='cpu').view(-1, 1, 1) * num_points

    # print("num_points", num_points)
    # print("x=", x.shape)
    # print("idx_base=", idx_base.shape)
    # print("idx_base=", idx_base)

    idx = idx + idx_base
    # print("idx",idx.shape)
    idx = idx.view(-1)
    # print("idx",idx.shape)
    _, num_dims, _ = x.size()
    # print("num_dims=",num_dims)

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    # print("feature=", feature.shape)
    # print("feature=",feature)

    return feature


class model_own(nn.Module):
    # args= Namespace(batch_size=32, Datasets='modelnet40', dropout=0.5, emb_dims=1024, epochs=250, eval=False,
    # exp_name='exp', k=20, lr=0.001, model='dgcnn', model_path='', momentum=0.9, no_cuda=False, num_points=1024, seed=1, test_batch_size=16, use_sgd=True)
    #
    # input:
    #       point:[B,C,N]
    # return:
    #       feature:[B,256]
    #
    # def __init__(self, args, output_channels=1):
    def __init__(self, k=20, dropout=0.5, output_channels=256):
        super(model_own, self).__init__()

        # self.args = args
        self.k = k

        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.bn5 = torch.nn.BatchNorm2d(512)
        self.bn6 = torch.nn.BatchNorm2d(1028)
        # 用于训练特征提取网络的损失函数
        self.bn_down_1 = torch.nn.BatchNorm2d(64)
        self.bn_down_2 = torch.nn.BatchNorm2d(64)
        self.bn_down_3 = torch.nn.BatchNorm2d(64)
        self.bn_down_4 = torch.nn.BatchNorm2d(64)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 32, kernel_size=1, bias=False),
            self.bn1,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d((3 + 32) * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d((3 + 32 + 64) * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            torch.nn.Conv2d((3 + 32 + 64 + 128) * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            torch.nn.Conv2d((3 + 32 + 64 + 128 + 256) * 2, 512, kernel_size=1, bias=False),
            self.bn5,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv6 = nn.Sequential(
            torch.nn.Conv2d((3 + 32 + 64 + 128 + 256 + 512) * 2, 1028, kernel_size=1, bias=False),
            self.bn6,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        self.pool = nn.MaxPool1d(40, stride=40)
        # self.linear1 = nn.Linear( (128+256+512)*1024, 512, bias=False)
        # self.linear1 = nn.Linear((128 + 256 + 512 + 1028) * 4, 512, bias=False)
        self.linear1 = nn.Linear((128 + 256 + 512 + 1028), 512, bias=False)
        self.bn7 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=args.dropout)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn8 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.conv_down_1 = nn.Sequential(
            torch.nn.Conv2d(32 * 2, 64, kernel_size=1, bias=False),
            self.bn_down_1,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_down_2 = nn.Sequential(
            torch.nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn_down_2,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_down_3 = nn.Sequential(
            torch.nn.Conv2d(128 * 2, 64, kernel_size=1, bias=False),
            self.bn_down_3,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_down_4 = nn.Sequential(
            torch.nn.Conv2d(256 * 2, 64, kernel_size=1, bias=False),
            self.bn_down_4,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        # self.linear1_down = nn.Linear(256*2, 256, bias=True)
        # self.linear2_down = nn.Linear(256, 128, bias=True)
        self.linear2_down = nn.Linear(128, 128, bias=True)
        self.linear3_down = nn.Linear(128, 64, bias=True)
        # self.linear4_down = nn.Linear(64, 6, bias=True)
        self.linear4_down = nn.Linear(64, 11, bias=True)  # 1017号使用

    def forward(self, x):
        device = torch.device('cuda:0')
        x = x.to(device)
        batch_size = x.size(0)
        # print("x.shape=",x.shape)
        x1_gf = get_graph_feature(x, k=self.k)
        # print("x1_gf.shape=",x1_gf.shape)
        x1_conv = self.conv1(x1_gf)
        # print("x1_conv.shape=", x1_conv.shape)
        x1_max = x1_conv.max(dim=-1, keepdim=False)[0]
        # print("x1_max.shape=", x1_max.shape)
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()

        x2_gf = get_graph_feature(x1_max, k=self.k)
        # print("x2_gf.shape=", x2_gf.shape)
        x2_gf = torch.cat((x1_gf, x2_gf), dim=1)
        # print("x2_gf.shape=", x2_gf.shape)
        x2_conv = self.conv2(x2_gf)
        # print("x2_conv.shape=", x2_conv.shape)
        x2_max = x2_conv.max(dim=-1, keepdim=False)[0]
        # print("x2_max.shape=", x2_max.shape)
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()

        # # 05.22做对比试验注释
        # x3_gf = get_graph_feature(x2_max, k=self.k)
        # #print("x3_gf.shape=", x3_gf.shape)
        # x3_gf=torch.cat((x2_gf,x3_gf),dim=1)
        # #print("x3_gf.shape=", x3_gf.shape)
        # x3_conv = self.conv3(x3_gf)
        # #print("x3_conv.shape=", x3_conv.shape)
        # x3_max = x3_conv.max(dim=-1, keepdim=False)[0]
        # # print("x3_max.shape=", x3_max.shape)
        # # if hasattr(torch.cuda, 'empty_cache'):
        # #     torch.cuda.empty_cache()
        #
        # x4_gf = get_graph_feature(x3_max, k=self.k)
        # #print("x4_gf.shape=", x4_gf.shape)
        # x4_gf=torch.cat((x3_gf,x4_gf),dim=1)
        # #print("x4_gf.shape=", x4_gf.shape)
        # x4_conv = self.conv4(x4_gf)
        # #print("x4_conv.shape=", x4_conv.shape)
        # x4_max = x4_conv.max(dim=-1, keepdim=False)[0]
        # #print("x4_max.shape=", x4_max.shape)
        # # if hasattr(torch.cuda, 'empty_cache'):
        # #     torch.cuda.empty_cache()

        # x5_gf = get_graph_feature(x4_max, k=self.k)
        # #print("x5_gf.shape=", x5_gf.shape)
        # x5_gf = torch.cat((x4_gf, x5_gf), dim=1)
        # #print("x5_gf.shape=", x5_gf.shape)
        # x5_conv = self.conv5(x5_gf)
        # #print("x5_conv.shape=", x5_conv.shape)
        # x5_max = x5_conv.max(dim=-1, keepdim=False)[0]
        # #print("x5_max.shape=", x5_max.shape)

        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()

        # x6_gf = get_graph_feature(x5_max, k=self.k)
        # #print("x.shape=", x6_gf.shape)
        # x6_gf = torch.cat((x5_gf, x6_gf), dim=1)
        # #print("x6_gf.shape=", x6_gf.shape)
        # x6_conv = self.conv6(x6_gf)
        # #print("x6.shape=", x6_conv.shape)
        # x6_max = x6_conv.max(dim=-1, keepdim=False)[0]
        # #print("x6_max.shape=", x6_max.shape)
        # # if hasattr(torch.cuda, 'empty_cache'):
        # #     torch.cuda.empty_cache()

        # 采用边缘卷积降维
        x_down_1 = get_graph_feature(x1_max, k=self.k)
        # print('x_down_1.shape',x_down_1.shape)
        x_down_1 = self.conv_down_1(x_down_1)
        x_down_1 = x_down_1.max(dim=-1, keepdim=False)[0]
        # print('x_down_1.shape',x_down_1.shape)

        x_down_2 = get_graph_feature(x2_max, k=self.k)
        x_down_2 = self.conv_down_2(x_down_2)
        x_down_2 = x_down_2.max(dim=-1, keepdim=False)[0]
        # print('x_down_2.shape', x_down_2.shape)

        # #05.22做对比试验注释
        # x_down_3 = get_graph_feature(x3_max, k=self.k)
        # x_down_3 = self.conv_down_3(x_down_3)
        # x_down_3 = x_down_3.max(dim=-1, keepdim=False)[0]
        # print('x_down_3.shape', x_down_3.shape)
        #
        # x_down_4 = get_graph_feature(x4_max, k=self.k)
        # x_down_4 = self.conv_down_4(x_down_4)
        # x_down_4 = x_down_4.max(dim=-1, keepdim=False)[0]
        # print('x_down_4.shape', x_down_4.shape)

        # x = torch.cat((x_down_1, x_down_2, x_down_3, x_down_4), dim=1)
        x = torch.cat((x_down_1, x_down_2), dim=1)
        # print("x.shape=",x.shape)
        x = x.view(1, -1, 128)
        print("x.view.shape=", x.shape)
        # x = x.view(1, -1, 256)
        # # print("x.shape=", x.shape)
        x = x.mean(axis=1, keepdim=False)
        print("x.shape=", x.shape)
        # x_max_pool = F.adaptive_max_pool1d(x,1).view(batch_size,-1)
        # x_avg_pool = F.adaptive_avg_pool1d(x,1).view(batch_size,-1)
        # x = torch.cat((x_max_pool, x_avg_pool), dim=1)
        # x = F.leaky_relu(self.linear1_down(x), negative_slope=0.2)
        # x = torch.nn.Dropout(x)
        x = F.leaky_relu(self.linear2_down(x), negative_slope=0.2)
        # x = torch.nn.Dropout(x)
        x = F.leaky_relu(self.linear3_down(x), negative_slope=0.2)
        x_classify = F.leaky_relu(self.linear4_down(x), negative_slope=0.2)
        # x_classify = torch.nn.Dropout(x_classify)
        print('x_classify.shape=', x_classify.shape)
        # x_classify=F.softmax(x_classify)
        print('x_classify=', x_classify)

        # 原始采用全连接层和池化降维
        # x = torch.cat((x3_max, x4_max, x5_max, x6_max), dim=1)
        # #print("x.shape=",x.shape)
        # #x=self.pool(x) #此处为减少MLP层输入节点数，后续在2D卷积部分增加步长来替换此步骤
        # #x=x.view(1,-1,1924)
        # x=x.mean(axis=2,keepdim=False)
        #
        # #print("x.shape=",x.shape)
        #
        # x=self.linear1(x)
        # #print("x_linaer1.shape=", x.shape)
        #
        # # x=self.bn7(x)
        # # print("x_bn7.shape=", x.shape) #batch_size=1的时候，无法使用batchmormal，后续使用patch输入的时候再重新使用
        # # 原因是模型中含有nn.BatchNorm层，训练时需要batch_size大于1，来计算当前batch的running mean and std。
        #
        # x=F.leaky_relu(x, negative_slope=0.2)
        #
        # #x=F.leaky_relu(self.bn7(self.linear1(x)), negative_slope=0.2)
        # x=self.dp1(x)
        # # x=F.leaky_relu(self.bn8(self.linear2(x)), negative_slope=0.2)
        # x = F.leaky_relu(self.linear2(x), negative_slope=0.2)
        # x=self.dp2(x)
        # #print("x_dp2.shape=", x.shape)
        # x=self.linear3(x)
        # #print("x=",x)
        # print("x_GraphFeature.shape=",x.shape)

        return x_classify


class model_own_tune(nn.Module):
    # args= Namespace(batch_size=32, Datasets='modelnet40', dropout=0.5, emb_dims=1024, epochs=250, eval=False,
    # exp_name='exp', k=20, lr=0.001, model='dgcnn', model_path='', momentum=0.9, no_cuda=False, num_points=1024, seed=1, test_batch_size=16, use_sgd=True)
    #
    # input:
    #       point:[B,C,N]
    # return:
    #       feature:[B,256]
    #
    # def __init__(self, args, output_channels=1):
    def __init__(self, k=20, dropout=0.5, output_channels=256):
        super(model_own_tune, self).__init__()

        # self.args = args
        self.k = k

        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.bn5 = torch.nn.BatchNorm2d(512)
        self.bn6 = torch.nn.BatchNorm2d(1028)
        # 用于训练特征提取网络的损失函数
        self.bn_down_1 = torch.nn.BatchNorm2d(64)
        self.bn_down_2 = torch.nn.BatchNorm2d(64)
        self.bn_down_3 = torch.nn.BatchNorm2d(64)
        self.bn_down_4 = torch.nn.BatchNorm2d(64)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 32, kernel_size=1, bias=False),
            self.bn1,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d((3 + 32) * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d((3 + 32 + 64) * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            torch.nn.Conv2d((3 + 32 + 64 + 128) * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            torch.nn.Conv2d((3 + 32 + 64 + 128 + 256) * 2, 512, kernel_size=1, bias=False),
            self.bn5,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv6 = nn.Sequential(
            torch.nn.Conv2d((3 + 32 + 64 + 128 + 256 + 512) * 2, 1028, kernel_size=1, bias=False),
            self.bn6,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        self.pool = nn.MaxPool1d(40, stride=40)
        # self.linear1 = nn.Linear( (128+256+512)*1024, 512, bias=False)
        # self.linear1 = nn.Linear((128 + 256 + 512 + 1028) * 4, 512, bias=False)
        self.linear1 = nn.Linear((128 + 256 + 512 + 1028), 512, bias=False)
        self.bn7 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=args.dropout)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn8 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.conv_down_1 = nn.Sequential(
            torch.nn.Conv2d(32 * 2, 64, kernel_size=1, bias=False),
            self.bn_down_1,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_down_2 = nn.Sequential(
            torch.nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn_down_2,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_down_3 = nn.Sequential(
            torch.nn.Conv2d(128 * 2, 64, kernel_size=1, bias=False),
            self.bn_down_3,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_down_4 = nn.Sequential(
            torch.nn.Conv2d(256 * 2, 64, kernel_size=1, bias=False),
            self.bn_down_4,
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
        # self.linear1_down = nn.Linear(256*2, 256, bias=True)
        # self.linear2_down = nn.Linear(256, 128, bias=True)
        self.linear2_down = nn.Linear(128, 128, bias=True)
        self.linear3_down = nn.Linear(128, 64, bias=True)
        # self.linear4_down = nn.Linear(64, 6, bias=True)
        self.linear4_down = nn.Linear(64, 11, bias=True)  # 1017号使用

    def forward(self, x):
        b = x.shape[0]
        device = torch.device('cuda:0')
        x = x.to(device)
        batch_size = x.size(0)
        # print("x.shape=",x.shape)
        x1_gf = get_graph_feature(x, k=self.k)
        # print("x1_gf.shape=",x1_gf.shape)
        x1_conv = self.conv1(x1_gf)
        # print("x1_conv.shape=", x1_conv.shape)
        x1_max = x1_conv.max(dim=-1, keepdim=False)[0]
        # print("x1_max.shape=", x1_max.shape)
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()

        x2_gf = get_graph_feature(x1_max, k=self.k)
        # print("x2_gf.shape=", x2_gf.shape)
        x2_gf = torch.cat((x1_gf, x2_gf), dim=1)
        # print("x2_gf.shape=", x2_gf.shape)
        x2_conv = self.conv2(x2_gf)
        # print("x2_conv.shape=", x2_conv.shape)
        x2_max = x2_conv.max(dim=-1, keepdim=False)[0]
        # print("x2_max.shape=", x2_max.shape)
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()

        # # 05.22做对比试验注释
        # x3_gf = get_graph_feature(x2_max, k=self.k)
        # #print("x3_gf.shape=", x3_gf.shape)
        # x3_gf=torch.cat((x2_gf,x3_gf),dim=1)
        # #print("x3_gf.shape=", x3_gf.shape)
        # x3_conv = self.conv3(x3_gf)
        # #print("x3_conv.shape=", x3_conv.shape)
        # x3_max = x3_conv.max(dim=-1, keepdim=False)[0]
        # # print("x3_max.shape=", x3_max.shape)
        # # if hasattr(torch.cuda, 'empty_cache'):
        # #     torch.cuda.empty_cache()
        #
        # x4_gf = get_graph_feature(x3_max, k=self.k)
        # #print("x4_gf.shape=", x4_gf.shape)
        # x4_gf=torch.cat((x3_gf,x4_gf),dim=1)
        # #print("x4_gf.shape=", x4_gf.shape)
        # x4_conv = self.conv4(x4_gf)
        # #print("x4_conv.shape=", x4_conv.shape)
        # x4_max = x4_conv.max(dim=-1, keepdim=False)[0]
        # #print("x4_max.shape=", x4_max.shape)
        # # if hasattr(torch.cuda, 'empty_cache'):
        # #     torch.cuda.empty_cache()

        # x5_gf = get_graph_feature(x4_max, k=self.k)
        # #print("x5_gf.shape=", x5_gf.shape)
        # x5_gf = torch.cat((x4_gf, x5_gf), dim=1)
        # #print("x5_gf.shape=", x5_gf.shape)
        # x5_conv = self.conv5(x5_gf)
        # #print("x5_conv.shape=", x5_conv.shape)
        # x5_max = x5_conv.max(dim=-1, keepdim=False)[0]
        # #print("x5_max.shape=", x5_max.shape)

        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()

        # x6_gf = get_graph_feature(x5_max, k=self.k)
        # #print("x.shape=", x6_gf.shape)
        # x6_gf = torch.cat((x5_gf, x6_gf), dim=1)
        # #print("x6_gf.shape=", x6_gf.shape)
        # x6_conv = self.conv6(x6_gf)
        # #print("x6.shape=", x6_conv.shape)
        # x6_max = x6_conv.max(dim=-1, keepdim=False)[0]
        # #print("x6_max.shape=", x6_max.shape)
        # # if hasattr(torch.cuda, 'empty_cache'):
        # #     torch.cuda.empty_cache()

        # 采用边缘卷积降维
        x_down_1 = get_graph_feature(x1_max, k=self.k)
        # print('x_down_1.shape',x_down_1.shape)
        x_down_1 = self.conv_down_1(x_down_1)
        x_down_1 = x_down_1.max(dim=-1, keepdim=False)[0]
        # print('x_down_1.shape',x_down_1.shape)

        x_down_2 = get_graph_feature(x2_max, k=self.k)
        x_down_2 = self.conv_down_2(x_down_2)
        x_down_2 = x_down_2.max(dim=-1, keepdim=False)[0]
        # print('x_down_2.shape', x_down_2.shape)

        # #05.22做对比试验注释
        # x_down_3 = get_graph_feature(x3_max, k=self.k)
        # x_down_3 = self.conv_down_3(x_down_3)
        # x_down_3 = x_down_3.max(dim=-1, keepdim=False)[0]
        # print('x_down_3.shape', x_down_3.shape)
        #
        # x_down_4 = get_graph_feature(x4_max, k=self.k)
        # x_down_4 = self.conv_down_4(x_down_4)
        # x_down_4 = x_down_4.max(dim=-1, keepdim=False)[0]
        # print('x_down_4.shape', x_down_4.shape)

        # x = torch.cat((x_down_1, x_down_2, x_down_3, x_down_4), dim=1)
        x = torch.cat((x_down_1, x_down_2), dim=1)
        # print("x.shape=",x.shape)
        x = x.view(b, -1, 128)
        # x = x.view(1, -1, 256)
        # # print("x.shape=", x.shape)
        x = x.mean(axis=1, keepdim=False)
        return x
        # print("x.shape=", x.shape)
        # x_max_pool = F.adaptive_max_pool1d(x,1).view(batch_size,-1)
        # x_avg_pool = F.adaptive_avg_pool1d(x,1).view(batch_size,-1)
        # x = torch.cat((x_max_pool, x_avg_pool), dim=1)
        # x = F.leaky_relu(self.linear1_down(x), negative_slope=0.2)
        # x = torch.nn.Dropout(x)
        x = F.leaky_relu(self.linear2_down(x), negative_slope=0.2)
        # x = torch.nn.Dropout(x)
        x = F.leaky_relu(self.linear3_down(x), negative_slope=0.2)
        x_classify = F.leaky_relu(self.linear4_down(x), negative_slope=0.2)
        # x_classify = torch.nn.Dropout(x_classify)
        print('x_classify.shape=', x_classify.shape)
        # x_classify=F.softmax(x_classify)
        print('x_classify=', x_classify)

        # 原始采用全连接层和池化降维
        # x = torch.cat((x3_max, x4_max, x5_max, x6_max), dim=1)
        # #print("x.shape=",x.shape)
        # #x=self.pool(x) #此处为减少MLP层输入节点数，后续在2D卷积部分增加步长来替换此步骤
        # #x=x.view(1,-1,1924)
        # x=x.mean(axis=2,keepdim=False)
        #
        # #print("x.shape=",x.shape)
        #
        # x=self.linear1(x)
        # #print("x_linaer1.shape=", x.shape)
        #
        # # x=self.bn7(x)
        # # print("x_bn7.shape=", x.shape) #batch_size=1的时候，无法使用batchmormal，后续使用patch输入的时候再重新使用
        # # 原因是模型中含有nn.BatchNorm层，训练时需要batch_size大于1，来计算当前batch的running mean and std。
        #
        # x=F.leaky_relu(x, negative_slope=0.2)
        #
        # #x=F.leaky_relu(self.bn7(self.linear1(x)), negative_slope=0.2)
        # x=self.dp1(x)
        # # x=F.leaky_relu(self.bn8(self.linear2(x)), negative_slope=0.2)
        # x = F.leaky_relu(self.linear2(x), negative_slope=0.2)
        # x=self.dp2(x)
        # #print("x_dp2.shape=", x.shape)
        # x=self.linear3(x)
        # #print("x=",x)
        # print("x_GraphFeature.shape=",x.shape)

        # return x_classify


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    a = torch.randn(40, 3, 2000).to(device)
    print(a.device)
    print(a)
    model = model_own().cuda()
    feature = model(a)
    print('feature', feature.shape)
