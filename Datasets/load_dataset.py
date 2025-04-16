# -*- coding: utf-8 -*-
"""
@author lizheng
@date 2022年12月01日 00:02:57
@packageName 
@className load_data
@version 1.0.0
@describe Done
"""
import glob
import itertools
from pympler import asizeof
import re

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
import time
import pathlib
import shutil
import traceback
from tqdm import tqdm

import torch
import sys
import os
import pandas as pd
from distortion import *
from FPS import farthest_point_sample, index_points, query_ball_point, knn, fps_sampling_func

# from open3d.cpu.pybind.geometry import PointCloud

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Auto_Distortion:
    def __init__(self, level_len: int) -> object:
        self.distortion_type_label = []
        self.distortion_level_label = []

        self.distortion_type = ["GN", "UN", "IN", "EN", "RD", "GD", "OC"]  #
        self.distortion_type_len = len(self.distortion_type)
        self.distortion_level_len = level_len
        self.distortion_level = {
            "GN": np.linspace(0.1, 0.7, self.distortion_level_len),
            "UN": np.linspace(0.3, 2.1, self.distortion_level_len),
            "IN": np.linspace(0.3, 2.1, self.distortion_level_len),
            "EN": np.linspace(0.1, 0.7, self.distortion_level_len),
            # "RD": np.linspace(0.15, 0.7, self.distortion_level_len),
            # "GD": np.linspace(1.2, 2.5, self.distortion_level_len),
            # "OC": np.linspace(0.01, 0.025, self.distortion_level_len),
        }

    def set_distortion(self, distortion_type_label, distortion_level_label):
        """
        设置增强标签，此函数应当在类实例化后进行设置增强标签
        :param distortion_type_label:   List of distortion_type_label
        :param distortion_level_label:  List of distortion_level_label
        :return:    No return
        """
        self.distortion_type_label = distortion_type_label
        self.distortion_level_label = distortion_level_label

    def auto_data_enhancement(self, pc, distortion_type_idx, distortion_level_idx, average_edge_l):
        pc_with_noise = None
        print(distortion_type_idx)
        print(self.distortion_type[distortion_type_idx])
        print(distortion_level_idx - 1)
        noise_level = self.distortion_level[self.distortion_type[distortion_type_idx]][
            distortion_level_idx - 1]
        if distortion_type_idx == 0:
            pc_with_noise = add_gauss_noise(pc, mean=0, std=noise_level * average_edge_l)
        elif distortion_type_idx == 1:
            pc_with_noise = add_uniform_noise(pc, noise_level * average_edge_l)
        elif distortion_type_idx == 2:
            pc_with_noise = add_impulse_noise(pc, noise_level * average_edge_l)
        elif distortion_type_idx == 3:
            pc_with_noise = add_exponent_noise(pc, noise_level * average_edge_l)
        elif distortion_type_idx == 4:
            pc_with_noise = pc_random_downsample(pc, noise_level)
        elif distortion_type_idx == 5:
            pc_with_noise = pc_grid_downsample(pc, noise_level * average_edge_l)
        elif distortion_type_idx == 6:
            pc_with_noise = pc_octree_compress(pc, noise_level, False)
        return pc_with_noise

    def add_geometry_noise(self, cnt, mini_patch):
        """
        对一个mini patch进行几何噪声增强
        :param cnt: index of label
        :param mini_patch: mini patch(patch_size * 3)
        :return: mini patch with noise
        """
        points = mini_patch
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 计算平均边长
        average_edge_l = np.asarray(pcd.compute_nearest_neighbor_distance()).mean()

        mini_patch = self.auto_data_enhancement(points, self.distortion_type_label[cnt],
                                                self.distortion_level_label[cnt],
                                                average_edge_l)
        return mini_patch

    def add_downsample_noise(self, cnt, points):
        """
        对整个点云进行下采样增强
        :param cnt: index of label
        :param points:  all point cloud (N*3)
        :return:    all points cloud with downsample (n*3)
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # 计算平均边长
        average_edge_l = np.asarray(pcd.compute_nearest_neighbor_distance()).mean()

        noise_points = self.auto_data_enhancement(points, self.distortion_type_label[cnt],
                                                  self.distortion_level_label[cnt],
                                                  average_edge_l)
        return noise_points


'''
# 老版本对patch载入  载入单个点云
class Patch_Point_Cloud_Datasets(Dataset):
    def __init__(self, raw_root_path, dis_root_path, train_size, level_len, patch_num, patch_size, train=True,
                 random_split=False):
        """
        !!!!!注意 要使用随机分类测试样本功能 一定要先初始化 train
        :param raw_root_path: 原始数据集路径
        :param dis_root_path: 失真数据集路径
        :param train_size: 数据中训练集和测试集的比例
        :param level_len:
        :param patch_num:
        :param patch_size:
        :param train: 设置为训练模式随机分类样本为训练集还是测试集
        """
        self.train = train
        self.level_len = level_len
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.train_size = train_size
        self.root_path = raw_root_path
        self.dis_roo_path = dis_root_path
        self.model_list = os.listdir(self.root_path)
        self.len = len(self.model_list)

        self.train_model_list = []
        self.train_len = 0
        self.test_model_list = []
        self.test_len = 0

        if self.train and random_split:
            self.random_train_test_spilt()

        self.train_model_list = list(pd.read_csv(os.path.join(FILE_PATH, 'train_split.csv'))['filename'])
        self.test_model_list = list(pd.read_csv(os.path.join(FILE_PATH, 'test_split.csv'))['filename'])
        self.train_len = len(self.train_model_list)
        self.test_len = len(self.test_model_list)

        self.distortion_points_list = []

        test = False
        if test:
            print(self.test_model_list)

    def random_train_test_spilt(self):
        all_idx = np.arange(self.len)
        train_idx = np.random.choice(all_idx, int(self.len * self.train_size), replace=False)
        test_idx = np.delete(all_idx, train_idx)
        self.train_len = len(train_idx)
        self.test_len = len(test_idx)

        [self.train_model_list.append(self.model_list[idx]) for idx in train_idx]
        [self.test_model_list.append(self.model_list[idx]) for idx in test_idx]

        train_pd = pd.DataFrame(self.train_model_list, columns=['filename'])
        test_pd = pd.DataFrame(self.test_model_list, columns=['filename'])

        train_pd.to_csv(os.path.join(BASE_DIR, 'train_split.csv'))
        test_pd.to_csv(os.path.join(BASE_DIR, 'test_split.csv'))

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len

    def fps_sampling(self, points):
        points = points.reshape((1, -1, 3))
        points = torch.from_numpy(points).to(torch.float32)
        # print(points.shape)
        # 对模型进行采样，取patch
        centroids_index = farthest_point_sample(points, self.patch_num)  # 每次采样点数
        centroids = index_points(points, centroids_index)  # centroids:[b s C]
        # print('centroids_index:', centroids_index)
        # radius采样
        result = torch.zeros((1, self.patch_num, self.patch_size), dtype=torch.int64)

        for i in range(self.patch_num):
            # # 方案一
            # dis_point_cloud = torch.from_numpy(
            #     np.asarray(o3d.io.read_point_cloud(self.distortion_points_list[i]).points))
            # 方案二
            dis_point_cloud = self.distortion_points_list[i]
            knn_idx = knn(centroids[:, 0, :], dis_point_cloud, k=1)
            search_point = dis_point_cloud[knn_idx[0]].reshape((1, 1, 3))
            search_point_cloud = dis_point_cloud.reshape((1, -1, 3))
            single_result = query_ball_point(0.2, self.patch_size, search_point_cloud,
                                             search_point)  # result:[b s n_sample]
            result[0, i, :] = single_result.squeeze()
        b, s, patch_size = result.shape
        out_data_tensor = torch.zeros((b, s, patch_size, 3), dtype=torch.float32)
        for patch in range(s):  # 0-64
            # # 方案一
            # dis_point_cloud = torch.from_numpy(
            #     np.asarray(o3d.io.read_point_cloud(self.distortion_points_list[patch]).points))
            # 方案二
            dis_point_cloud = self.distortion_points_list[patch]
            patch_index = result[:, patch, :]  # [b n_sample]，n_sample=patch_size
            search_point_cloud = dis_point_cloud.reshape((1, -1, 3))
            value = index_points(search_point_cloud, patch_index)  # value:[b patch_size C]
            for batch in range(b):
                out_data_tensor[batch][patch] = value[batch]  # result_value:[b s patch_size C],s*patch_size=N
        out_data_tensor = out_data_tensor.squeeze()
        return out_data_tensor

    def load_raw_points_from_file(self, idx_model, distortion_type_label, distortion_level_label):
        distortion_type_path = ["gaussian_noise", "uniform_noise", "impulse_noise", "index_noise",
                                "random_downsample", "gridAverage_downsample", "OctreeCom"]

        raw_path = 'raw_model_%d' % idx_model
        for i in range(self.patch_num):
            if distortion_level_label[i] == 0:
                distortion_level_path = raw_path + '.ply'
            else:
                distortion_level_path = 'level%d.ply' % (distortion_level_label[i])
            path = os.path.join(self.dis_roo_path,
                                raw_path,
                                distortion_type_path[distortion_type_label[i]],
                                distortion_level_path,
                                )
            # #  方案一 后期载入点云 内存消耗少 运算时间长
            # self.distortion_points_list.append(path)

            # 方案二 一次性加载点云 内存消耗多 运算速度快
            self.distortion_points_list.append(torch.from_numpy(
                np.asarray(o3d.io.read_point_cloud(path).points)))

    def __getitem__(self, item):
        if self.train:
            original_filename = self.train_model_list[item]
        else:
            original_filename = self.test_model_list[item]
        file_path = os.path.join(self.root_path, original_filename)

        # 随机生成增强标签，实例化数据增强对象
        auto_distortion = Auto_Distortion(level_len=self.level_len)

        # 随机生成标签
        distortion_type_label = np.random.randint(0, 7, self.patch_num)
        # distortion_level_label = np.random.randint(0, self.level_len + 1, self.patch_num)
        # todo 尝试生成默认等级标签
        distortion_level_label = np.ones(self.patch_num, dtype=np.int8) * 5  # 全部是level2标签

        auto_distortion.set_distortion(distortion_type_label, distortion_level_label)

        if self.train:
            model_idx = int(re.findall(r'\d+', self.train_model_list[item])[0])
        else:
            model_idx = int(re.findall(r'\d+', self.test_model_list[item])[0])
        self.load_raw_points_from_file(model_idx, distortion_type_label, distortion_level_label)
        # print('distortion_type_label:', distortion_type_label)
        # print('distortion_level_label:', distortion_level_label)

        # 载入点云和patch
        points = np.asarray(o3d.io.read_point_cloud(file_path).points)
        patch = self.fps_sampling(points)
        patch_with_noise = patch

        # # 现场生成失真
        # patch_with_noise = np.zeros_like(patch)
        #
        # for i in range(self.patch_num):
        #     if distortion_type_label[i] < 4:  # 注意增加失真类型需要更改此值   4 表示前四个是几何类型失真
        #         # 做几何失真
        #         mini_patch = patch[i, :, :]
        #         mini_patch = auto_distortion.add_geometry_noise(i, mini_patch)
        #         patch_with_noise[i, :, :] = mini_patch
        #     else:
        #         # 做下采样失真
        #         noise_points = auto_distortion.add_downsample_noise(i, points)
        #         noise_patch = self.fps_sampling(noise_points)
        #         mini_patch = noise_patch[i]
        #         patch_with_noise[i, :, :] = mini_patch

        return patch_with_noise, distortion_type_label, distortion_level_label
'''


def random_train_test_spilt(self):
    all_idx = np.arange(self.len)
    train_idx = np.random.choice(all_idx, int(self.len * self.train_size), replace=False)
    test_idx = np.delete(all_idx, train_idx)
    self.train_len = len(train_idx)
    self.test_len = len(test_idx)

    [self.train_model_list.append(self.model_list[idx]) for idx in train_idx]
    [self.test_model_list.append(self.model_list[idx]) for idx in test_idx]

    train_pd = pd.DataFrame(self.train_model_list, columns=['filename'])
    test_pd = pd.DataFrame(self.test_model_list, columns=['filename'])

    train_pd.to_csv(os.path.join(BASE_DIR, 'train_split.csv'))
    test_pd.to_csv(os.path.join(BASE_DIR, 'test_split.csv'))


class Pairwise_Patch_Point_Cloud_Datasets_Input_PLY(Dataset):
    def __init__(self, raw_root_path, dis_root_path, train_size, level_len,
                 patch_num=64, patch_size=512, multiplier=1,
                 train=True, random_split=False, reshape_model=True, pre_load=True):
        self.raw_root_path = raw_root_path
        self.dis_root_path = dis_root_path
        self.train_size = train_size
        self.level_len = level_len
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.multiplier = multiplier
        self.train = train
        self.reshape_model = reshape_model
        self.model_list = os.listdir(self.raw_root_path)
        self.len = len(self.model_list)
        self.train_model_list = []
        self.test_model_list = []
        self.train_len = 0
        self.test_len = 0

        self.distortion_type_list = ["noise1", "noise2", "noise3", "noise4", 'sd']

        # self.distortion_type_list = ["gaussian_noise", "uniform_noise", "impulse_noise", "index_noise",
        #                              "random_downsample", "gridAverage_downsample", "OctreeCom"]
        # self.distortion_type_list = ["gridAverage", "noise1", "noise2", "noise3",
        #                              "noise4", "OctreeCom", "random"]

        if self.train and random_split:
            random_train_test_spilt(self)

        self.train_model_list = list(
            pd.read_csv(os.path.join(FILE_PATH, 'random_split_dir', 'train_split.csv'))['filename'])
        self.test_model_list = list(
            pd.read_csv(os.path.join(FILE_PATH, 'random_split_dir', 'test_split.csv'))['filename'])
        self.train_len = len(self.train_model_list)
        self.test_len = len(self.test_model_list)

        self.pc2memory = {}
        self.noise2memory = {}
        if pre_load:
            self.preload2memory()

    def __len__(self):
        if self.train:
            return self.train_len * self.multiplier
        else:
            return self.test_len * self.multiplier

    def preload2memory(self):
        def human_readable_size(size_bytes):
            """
            将字节数转换为人类易读的格式
            :param size_bytes: 内存大小（字节）
            :return: 带单位的字符串
            """
            if size_bytes == 0:
                return "0 B"

            # 定义单位和转换比例
            size_name = ("B", "KB", "MB", "GB", "TB")
            i = int((len(str(size_bytes)) - 1) / 3) if size_bytes < 1024 else min(int(size_bytes.bit_length() / 10), 4)

            p = 1024 ** i
            s = round(size_bytes / p, 2)
            return f"{s} {size_name[i]}"
        model_list = self.train_model_list if self.train else self.test_model_list
        combinations = list(itertools.product(self.distortion_type_list, list(range(1, self.level_len + 1))))
        for i, model_idx in enumerate(tqdm(model_list, desc='Pre loading data')):
            ply_file_path = os.path.join(self.dis_root_path, model_idx[: -4], "%s.ply" % model_idx[: -4])
            points = torch.from_numpy(np.asarray(o3d.io.read_point_cloud(ply_file_path).points))
            self.pc2memory[model_idx] = {
                'gt': points,
            }
            diss = {
                f"{comp[0]}_level{comp[1]}": torch.from_numpy(np.asarray(
                    o3d.io.read_point_cloud(
                        os.path.join(self.dis_root_path, model_idx[: -4], comp[0], f"level{comp[1]}.ply")
                    ).points)
                )
                for comp in combinations
            }
            self.pc2memory[model_idx].update(diss)

            self.noise2memory[model_idx] = {
                key: value - self.pc2memory[model_idx]['gt'] for key, value in self.pc2memory[model_idx].items()
                if not key.startswith('sd')
            }

        total_size_bytes1 = asizeof.asizeof(self.pc2memory)
        total_size_bytes2 = asizeof.asizeof(self.noise2memory)
        print(f"Total size: {human_readable_size(total_size_bytes1 + total_size_bytes2)}")

    def __getitem__(self, item):
        if self.train:
            model_idx = self.train_model_list[item % self.train_len]
        else:
            model_idx = self.test_model_list[item % self.test_len]

        distortion_type_label = np.random.randint(0, 6)  # 生成失真类别标签
        alpha = torch.cat((torch.softmax(torch.rand(4), dim=0), torch.tensor([0.])))
        if distortion_type_label < 5:
            alpha = torch.zeros(5)
            alpha[distortion_type_label] = 1.

        # 0-3 represent GN, UN, IN, EN, and 4 represent SD 5 represent Mixed Noise
        distortion_level_label_list = np.arange(self.level_len + 1)
        np.random.shuffle(distortion_level_label_list)
        distortion_level_label_list = distortion_level_label_list[:2]
        pairwise_label = 1 if distortion_level_label_list[0] < distortion_level_label_list[1] else 0
        pairwise_label = torch.tensor([pairwise_label])

        if distortion_type_label < 5:
            pth_dis_dir = os.path.join(self.dis_root_path, model_idx[: -4],
                                       self.distortion_type_list[distortion_type_label])
        data_pth_list = []
        ply_file_path = ''
        centroids = None
        # or_ply_file_path = os.path.join(self.dis_root_path, model_idx[: -4], "%s.ply" % model_idx[: -4])
        # or_points = torch.from_numpy(np.asarray(o3d.io.read_point_cloud(or_ply_file_path).points))
        or_points = self.pc2memory[model_idx]['gt']
        _, centroids = fps_sampling_func(or_points, self.patch_num, self.patch_size, centroids)
        for distortion_level_label in distortion_level_label_list:

            if distortion_level_label == 0:
                # Ground Truth
                # ply_file_path = os.path.join(self.dis_root_path, model_idx[: -4], "%s.ply" % model_idx[: -4])
                # points = torch.from_numpy(np.asarray(o3d.io.read_point_cloud(ply_file_path).points))
                points = self.pc2memory[model_idx]['gt']
                points_with_patch, centroids = fps_sampling_func(points, self.patch_num, self.patch_size, centroids)
                data_pth_list.append(points_with_patch)
            else:
                if distortion_type_label < 5:
                    # ply_file_path = os.path.join(pth_dis_dir, "level%s.ply" % str(distortion_level_label))
                    # points = torch.from_numpy(np.asarray(o3d.io.read_point_cloud(ply_file_path).points))
                    points = \
                        self.pc2memory[model_idx][
                            f"{self.distortion_type_list[distortion_type_label]}_level{distortion_level_label}"]
                    points_with_patch, centroids = fps_sampling_func(points, self.patch_num, self.patch_size, centroids)
                    data_pth_list.append(points_with_patch)
                else:  # 混合噪声
                    noise_dict = {}
                    for i in range(4):
                        # pth_dis_dir = os.path.join(self.dis_root_path, model_idx[: -4], self.distortion_type_list[i])
                        # ply_file_path = os.path.join(pth_dis_dir, "level%s.ply" % str(distortion_level_label))
                        # points = torch.from_numpy(np.asarray(o3d.io.read_point_cloud(ply_file_path).points))
                        noise = self.noise2memory[model_idx][
                            f"{self.distortion_type_list[i]}_level{distortion_level_label}"
                        ]
                        weight_noise = alpha[i] * noise
                        noise_dict[i] = weight_noise
                    points = or_points + noise_dict[0] + noise_dict[1] + noise_dict[2] + noise_dict[3]
                    points_with_patch, centroids = fps_sampling_func(points, self.patch_num, self.patch_size, centroids)
                    data_pth_list.append(points_with_patch)

        return data_pth_list, distortion_type_label, distortion_level_label_list, pairwise_label, alpha


def tool_file_format(raw_dataset_path):
    raw_dataset_path = pathlib.Path(raw_dataset_path)
    model_names = [item for item in os.listdir(raw_dataset_path) if (raw_dataset_path / item).is_dir()]
    sub_folder = ['noise1', 'noise2', 'noise3', 'noise4', 'sd']

    for name in model_names:
        src = raw_dataset_path / name / f"{name}.ply"

        for sub in sub_folder:
            dst = raw_dataset_path / name / sub / f"{name}.ply"
            shutil.copy(src, dst)


"""class Pairwise_Patch_Point_Cloud_Datasets(Dataset):
    def __init__(self, raw_root_path, dis_root_path, train_size, level_len,
                 patch_num=64, patch_size=512,
                 train=True, random_split=False, reshape_model=True):
        self.raw_root_path = raw_root_path
        self.dis_root_path = dis_root_path
        self.train_size = train_size
        self.level_len = level_len
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.train = train
        self.reshape_model = reshape_model
        self.model_list = os.listdir(self.raw_root_path)
        self.len = len(self.model_list)
        self.train_model_list = []
        self.test_model_list = []
        self.train_len = 0
        self.test_len = 0
        self.distortion_type_list = ["gaussian_noise", "uniform_noise", "impulse_noise", "index_noise",
                                     "random_downsample", "gridAverage_downsample", "OctreeCom"]
        if self.train and random_split:
            self.random_train_test_spilt()

        self.train_model_list = list(
            pd.read_csv(os.path.join(FILE_PATH, 'random_split_dir', 'train_split.csv'))['filename'])
        self.test_model_list = list(
            pd.read_csv(os.path.join(FILE_PATH, 'random_split_dir', 'test_split.csv'))['filename'])
        self.train_len = len(self.train_model_list)
        self.test_len = len(self.test_model_list)

    def random_train_test_spilt(self):
        all_idx = np.arange(self.len)
        train_idx = np.random.choice(all_idx, int(self.len * self.train_size), replace=False)
        test_idx = np.delete(all_idx, train_idx)
        self.train_len = len(train_idx)
        self.test_len = len(test_idx)

        [self.train_model_list.append(self.model_list[idx]) for idx in train_idx]
        [self.test_model_list.append(self.model_list[idx]) for idx in test_idx]

        train_pd = pd.DataFrame(self.train_model_list, columns=['filename'])
        test_pd = pd.DataFrame(self.test_model_list, columns=['filename'])

        train_pd.to_csv(os.path.join(BASE_DIR, 'train_split.csv'))
        test_pd.to_csv(os.path.join(BASE_DIR, 'test_split.csv'))

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len

    def __getitem__(self, item):
        if self.train:
            model_idx = self.train_model_list[item]
        else:
            model_idx = self.test_model_list[item]

        distortion_type_label = np.random.randint(0, 7)  # 生成失真类别标签
        distortion_level_label_list = np.arange(self.level_len + 1)
        np.random.shuffle(distortion_level_label_list)
        distortion_level_label_list = distortion_level_label_list[:2]
        pairwise_label = 1 if distortion_level_label_list[0] < distortion_level_label_list[1] else 0
        pairwise_label = torch.tensor([pairwise_label])
        # print(distortion_level_label_list)
        # print('pairwise_label', pairwise_label)

        pth_dis_dir = os.path.join(self.dis_root_path, model_idx[: -4],
                                   self.distortion_type_list[distortion_type_label])

        data_pth_list = []
        for distortion_level_label in distortion_level_label_list:
            if distortion_level_label == 0:
                ply_file_path = os.path.join(pth_dis_dir, "%s.pth" % model_idx[: -4])
            else:
                ply_file_path = os.path.join(pth_dis_dir, "level%s.pth" % str(distortion_level_label))
            # print(distortion_level_label, "load file:", ply_file_path)
            data = torch.load(ply_file_path)  # type: torch.Tensor
            if self.reshape_model:
                data = data.permute(2, 0, 1).view(3, -1)
            # print('data.shape', data.shape)
            data_pth_list.append(data)
        return data_pth_list, distortion_type_label, distortion_level_label_list, pairwise_label"""


class Patch_Point_Cloud_Datasets(Dataset):
    """
    导入分patch后重组成点云的数据集(自监督)
    """

    def __init__(self, raw_root_path, dis_root_path, train_size, level_len,
                 patch_num=64, patch_size=512,
                 train=True, random_split=False):
        self.raw_root_path = raw_root_path
        self.dis_root_path = dis_root_path
        self.train_size = train_size
        self.level_len = level_len
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.train = train
        self.model_list = os.listdir(self.raw_root_path)
        self.len = len(self.model_list)
        self.train_model_list = []
        self.test_model_list = []
        self.train_len = 0
        self.test_len = 0
        self.distortion_type_list = ["gaussian_noise", "uniform_noise", "impulse_noise", "index_noise",
                                     "random_downsample", "gridAverage_downsample", "OctreeCom"]
        if self.train and random_split:
            self.random_train_test_spilt()

        self.train_model_list = list(
            pd.read_csv(os.path.join(FILE_PATH, 'random_split_dir', 'train_split.csv'))['filename'])
        self.test_model_list = list(
            pd.read_csv(os.path.join(FILE_PATH, 'random_split_dir', 'test_split.csv'))['filename'])
        self.train_len = len(self.train_model_list)
        self.test_len = len(self.test_model_list)

        self.pre_train_dis_level_flag = 0  # 用于预训练逐步提升失真等级情况

    def random_train_test_spilt(self):
        all_idx = np.arange(self.len)
        train_idx = np.random.choice(all_idx, int(self.len * self.train_size), replace=False)
        test_idx = np.delete(all_idx, train_idx)
        self.train_len = len(train_idx)
        self.test_len = len(test_idx)

        [self.train_model_list.append(self.model_list[idx]) for idx in train_idx]
        [self.test_model_list.append(self.model_list[idx]) for idx in test_idx]

        train_pd = pd.DataFrame(self.train_model_list, columns=['filename'])
        test_pd = pd.DataFrame(self.test_model_list, columns=['filename'])

        train_pd.to_csv(os.path.join(BASE_DIR, 'train_split.csv'))
        test_pd.to_csv(os.path.join(BASE_DIR, 'test_split.csv'))

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len

    def __getitem__(self, item):
        if self.train:
            model_idx = self.train_model_list[item]
        else:
            model_idx = self.test_model_list[item]

        # 随机生成标签
        # todo 测试label在
        distortion_type_label = np.random.randint(0, 7)
        distortion_level_label = np.random.randint(0, self.level_len + 1)
        # # todo 锁定level 为 5
        # distortion_level_label = 5
        # if self.pre_train_dis_level_flag == 0:

        #     distortion_level_label = 5
        # elif self.pre_train_dis_level_flag == 1:
        #     distortion_level_label = np.random.randint(4, 7)
        # elif self.pre_train_dis_level_flag == 2:
        #     distortion_level_label = np.random.randint(3, 8)
        # elif self.pre_train_dis_level_flag == 3:
        #     distortion_level_label = np.random.randint(1, self.level_len + 1)

        ply_file_path = os.path.join(self.dis_root_path, model_idx[: -4],
                                     self.distortion_type_list[distortion_type_label])

        print('ply_file_path:', ply_file_path)
        if distortion_level_label == 0:
            ply_file_path = os.path.join(ply_file_path, "%s.pth" % model_idx[: -4])
        else:
            ply_file_path = os.path.join(ply_file_path, "level%s.pth" % str(distortion_level_label))
        # print("load file:", ply_file_path)
        data = torch.load(ply_file_path)  # type: torch.Tensor
        data = data.permute(2, 0, 1).view(3, -1)
        return data, distortion_type_label, distortion_level_label


class No_Patch_Point_Cloud_Datasets(Dataset):
    def __init__(self, raw_root_path, dis_root_path, train_size, level_len, train=True, random_split=False):
        self.raw_root_path = raw_root_path
        self.dis_root_path = dis_root_path
        self.train_size = train_size
        self.level_len = level_len
        self.train = train
        self.model_list = os.listdir(self.raw_root_path)
        self.len = len(self.model_list)
        self.train_model_list = []
        self.test_model_list = []
        self.train_len = 0
        self.test_len = 0
        self.distortion_type_list = ["gaussian_noise", "uniform_noise", "impulse_noise", "index_noise",
                                     "random_downsample", "gridAverage_downsample", "OctreeCom"]

        if self.train and random_split:
            self.random_train_test_spilt()

        self.train_model_list = list(pd.read_csv(os.path.join(FILE_PATH, 'train_split.csv'))['filename'])
        self.test_model_list = list(pd.read_csv(os.path.join(FILE_PATH, 'test_split.csv'))['filename'])
        self.train_len = len(self.train_model_list)
        self.test_len = len(self.test_model_list)

    def random_train_test_spilt(self):
        all_idx = np.arange(self.len)
        train_idx = np.random.choice(all_idx, int(self.len * self.train_size), replace=False)
        test_idx = np.delete(all_idx, train_idx)
        self.train_len = len(train_idx)
        self.test_len = len(test_idx)

        [self.train_model_list.append(self.model_list[idx]) for idx in train_idx]
        [self.test_model_list.append(self.model_list[idx]) for idx in test_idx]

        train_pd = pd.DataFrame(self.train_model_list, columns=['filename'])
        test_pd = pd.DataFrame(self.test_model_list, columns=['filename'])

        train_pd.to_csv(os.path.join(BASE_DIR, 'train_split.csv'))
        test_pd.to_csv(os.path.join(BASE_DIR, 'test_split.csv'))

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len

    def __getitem__(self, item):
        if self.train:
            model_idx = self.train_model_list[item]
        else:
            model_idx = self.test_model_list[item]

        # 随机生成标签
        # todo 测试label在
        distortion_type_label = np.random.randint(0, 7)
        # distortion_level_label = np.random.randint(0, self.level_len + 1)
        # todo 锁定level 为 5
        distortion_level_label = 5

        ply_file_path = os.path.join(self.dis_root_path, model_idx[: -4],
                                     self.distortion_type_list[distortion_type_label])
        if distortion_level_label == 0:
            ply_file_path = os.path.join(ply_file_path, "%s.ply" % model_idx[: -4])
        else:
            ply_file_path = os.path.join(ply_file_path, "level%s.ply" % str(distortion_level_label))

        pcd = o3d.io.read_point_cloud(ply_file_path)
        points = np.asarray(pcd.points)

        points = torch.from_numpy(points)
        distortion_type_label = torch.from_numpy(np.array([distortion_type_label]))
        distortion_level_label = torch.from_numpy(np.array([distortion_level_label]))

        print(ply_file_path, distortion_type_label, distortion_level_label)

        return points, distortion_type_label, distortion_level_label


def create_preprocessing_datasets(raw_model_path, dis_model_path, levels):
    # raw_model_path = r'E:\homegate\R_Quality_Assessment\Dataset\Original_Database_data150'
    # dis_model_path = r'E:\homegate\R_Quality_Assessment\Dataset\tmp'
    pathlib.Path(dis_model_path).mkdir(parents=True, exist_ok=True)

    auto_distortion = Auto_Distortion(level_len=5)
    pc_files = glob.glob(os.path.join(raw_model_path, '*.ply'))
    noise_mapping = {
        'GN': 'noise1',
        'UN': 'noise2',
        'IN': 'noise3',
        'RD': 'noise4',
    }

    for pc_path in pc_files:
        pcd = o3d.io.read_point_cloud(pc_path)
        pts = np.asarray(pcd.points)
        average_edge_l = np.asarray(pcd.compute_nearest_neighbor_distance()).mean()
        for cnt, (noise_type, save_dir) in enumerate(noise_mapping.items()):
            for level in range(1, levels + 1):
                save_path = os.path.join(dis_model_path, os.path.basename(pc_path)[: -4], save_dir,
                                         'level%s.ply' % str(level))
                pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
                if os.path.exists(save_path):
                    continue
                noise_points = auto_distortion.auto_data_enhancement(pts,
                                                                     cnt,
                                                                     level,
                                                                     average_edge_l)
                noise_pcd = o3d.geometry.PointCloud()
                noise_pcd.points = o3d.utility.Vector3dVector(noise_points)
                o3d.io.write_point_cloud(save_path, noise_pcd)


def tools_pc_normalization(raw_dataset_path, dis_dataset_path):
    raw_dataset_path = pathlib.Path(raw_dataset_path)
    dis_dataset_path = pathlib.Path(dis_dataset_path)
    names = [item[:-4] for item in os.listdir(raw_dataset_path) if not item.startswith('raw')]
    print(names)
    sub_folder = ['noise1', 'noise2', 'noise3', 'noise4']
    for name in names:
        reference_pc_path = raw_dataset_path / f"{name}.ply"
        # pts = np.asarray(o3d.io.read_point_cloud(str(reference_pc_path)).points)

        dis_reference_path = dis_dataset_path / name / f"{name}.ply"
        dis_reference_pts = np.asarray(o3d.io.read_point_cloud(str(dis_reference_path)).points)
        min_point = np.min(dis_reference_pts, axis=0)
        dis_reference_pts -= min_point
        scale = np.max(dis_reference_pts)
        dis_reference_pts /= scale
        # shift point cloud center to the origin
        shift = np.mean(dis_reference_pts, axis=0)
        dis_reference_pts -= shift

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dis_reference_pts)
        o3d.io.write_point_cloud(str(dis_reference_path), pcd)

        for sub in sub_folder:
            pc_files = glob.glob(str(dis_dataset_path / name / sub / '*.ply'))
            pts_list = [np.asarray(o3d.io.read_point_cloud(pc_path).points) for pc_path in pc_files]
            pts_list = [(pts - min_point) / scale - shift for pts in pts_list]
            # write back
            pcd = o3d.geometry.PointCloud()
            for idx, pts in enumerate(pts_list):
                pcd.points = o3d.utility.Vector3dVector(pts)
                # o3d.io.write_point_cloud(pc_files[idx], pcd)

                print(pc_files)
                print(pts_list)
                break
        #
        # print(name)
        # # print(pts)
        # print(dis_reference_pts)
        # break

    # # pts smallest point shifts to the origin and scale normalization
    # min_point = np.min(pts, axis=0)
    # pts -= min_point
    # scale = np.max(pts)
    # pts /= scale
    # # shift point cloud center to the origin
    # pts -= np.mean(pts, axis=0)


if __name__ == '__main__':
    raw_dataset_path = r'/workspace/datasets/or150PStanford'
    dis_dataset_path = r'/workspace/datasets/or150PStanford_data5_v2'
    # alpha = 0.5
    training_epoches = 10 * 10
    epoch_multiplier = 105
    step_divisor = 630  # 630 为10%
    batch_size = 4
    patch_number = 64
    patch_size = 250
    theta = 0.5

    train_datasets = Pairwise_Patch_Point_Cloud_Datasets_Input_PLY(raw_dataset_path, dis_dataset_path, train_size=0.8,
                                                                   level_len=5, train=True, patch_size=patch_size,
                                                                   multiplier=epoch_multiplier, random_split=False)
    test_datasets = Pairwise_Patch_Point_Cloud_Datasets_Input_PLY(raw_dataset_path, dis_dataset_path, train_size=0.8,
                                                                  level_len=5, train=False, patch_size=patch_size,
                                                                  multiplier=epoch_multiplier, random_split=False)
    # data_pth_list, distortion_type_label, distortion_level_label_list, pairwise_label, alpha = train_datasets[0]


    def sample():
        try:
            data_pth_list, distortion_type_label, distortion_level_label_list, pairwise_label, alpha = train_datasets[0]
            print(distortion_type_label, alpha)
        except Exception:
            print(traceback.format_exc())


    breakpoint()

    # train_datasets.load_pc2memory()

    # breakpoint()

    #
    # raw_dataset_path = r'/workspace/datasets/Original_Database_data150'
    # dis_dataset_path = r'/workspace/datasets/data5_adaptive_denoising'
    #
    # # 分patch输入
    # train_datasets = Pairwise_Patch_Point_Cloud_Datasets_Input_PLY(raw_dataset_path, dis_dataset_path, train_size=0.8,
    #                                                                level_len=5, train=True, patch_size=256)
    # data_pth_list, distortion_type_label, distortion_level_label_list, pairwise_label = train_datasets[0]
    #
    # print(data_pth_list.__len__())
    # print(data_pth_list[0].shape)
    # print(os.getcwd())
    #
    # for step, item in enumerate(data_pth_list):
    #     points = item[:10, :, :].view(-1, 3).numpy()
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points)
    #     o3d.io.write_point_cloud('%d.ply' % step, pcd)

    # raw_model_path = r'E:\homegate\R_Quality_Assessment\Dataset\Original_Database_data150'
    # dis_model_path = r'E:\homegate\R_Quality_Assessment\Dataset\Data10_fixed_fps'
    #
    # # 分patch输入
    # train_datasets = Patch_Point_Cloud_Datasets(raw_model_path, dis_model_path, train_size=0.8,
    #                                             level_len=10, train=True)
    # test_datasets = Patch_Point_Cloud_Datasets(raw_model_path, dis_model_path, train_size=0.8,
    #                                            level_len=10, train=False)
    #
    # train_loader = DataLoader(train_datasets, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    #
    # train_datasets.pre_train_dis_level_flag = 1
    # for step, (x, y1, y2) in enumerate(train_loader):
    #     break
    #
    # print(x.shape)
    # print('y1', y1)
    # print('y2', y2)
    #
    # print("train_datasets", train_datasets.__len__())
    # print("test_datasets", test_datasets.__len__())
    #
    # print("train_loader", len(train_loader))

    # path = r'E:\homegate\R_Quality_Assessment\Dataset\Original_Database_data150\%s'
    # % train_datasets.train_model_list[0]
    # print(path)
    # or_pc = o3d.io.read_point_cloud(path)  # type:PointCloud
    # or_pc.paint_uniform_color([1, 1, 1])
    #
    # pc_tmp = x.numpy()[0, 63, :, :]
    # pc1 = o3d.geometry.PointCloud()
    # pc1.points = o3d.utility.Vector3dVector(pc_tmp)
    #
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    # render_option.background_color = np.array([1, 1, 1])  # 设置背景色
    # or_pc.paint_uniform_color([0, 1, 0])
    # vis.add_geometry(or_pc)  # 添加点云
    #
    # pc1.paint_uniform_color([1, 0, 0])
    # vis.add_geometry(pc1)  # 添加点云
    # vis.run()
    #
    # print(x.shape)

    # # 整个点云输入
    # train_datasets = No_Patch_Point_Cloud_Datasets(raw_model_path, dis_model_path, train_size=0.8, level_len=10,
    #                                                train=True)
    # train_loader = DataLoader(train_datasets, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # for step, (b_x, b_y1, b_y2) in enumerate(train_loader):
    #     if step > 10:
    #         break
    # print(b_x.shape)
    # print(b_y1)
    # print(b_y2)
