# -*- coding: utf-8 -*-
"""
@author lizheng
@date 2023年03月31日 16:27:57
@packageName 
@className load_tune_dataset
@version 1.0.0
@describe TODO
"""

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
import numpy as np
import open3d as o3d
from FPS import farthest_point_sample, index_points, query_ball_point, knn, fps_sampling_func


class PairwsiePatchPointCloudTunePLD(Dataset):
    def __init__(self, root_path, index_file, multiplier=1):
        self.f_line = ''
        self.patch_num = 64
        self.patch_size = 250
        self.multiplier = multiplier
        with open(os.path.join(root_path, index_file), 'r') as f:
            self.f_line = f.read().rstrip('\n').split('\n')
        self.len = self.f_line.__len__()

    def __len__(self):
        return self.f_line.__len__() * self.multiplier

    def __getitem__(self, item):
        idx = item % self.len
        file_path, pmos, = self.f_line[idx].split(' ')
        pmos = float(pmos)
        points = torch.from_numpy(np.asarray(o3d.io.read_point_cloud(file_path).points))
        points_with_patch = fps_sampling_func(points, self.patch_num, self.patch_size)
        return points_with_patch, pmos


class Pairwise_Patch_Point_Cloud_Tune_GPCD_Input(Dataset):
    def __init__(self, root_path, index_csv_path, file_suffix,
                 patch_num=64, patch_size=512, multiplier=1):
        self.root_path = root_path
        self.file_suffix = file_suffix
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.multiplier = multiplier
        self.d_list = [item for item in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, item))]
        self.d_list = sorted(self.d_list, key=lambda x: int(x[1:2]))

        self.index_df = pd.read_csv(index_csv_path)
        self.len = self.index_df.__len__()

    def __len__(self):
        return self.len * self.multiplier

    def __getitem__(self, item):
        idx = item % self.len
        if idx < 25:
            file_dir = os.path.join(self.root_path, self.d_list[0])
        else:
            file_dir = os.path.join(self.root_path, self.d_list[1])
        file_name = self.index_df.loc[idx, 'stimulus']
        label = self.index_df.loc[idx, 'MOS']

        data = None
        if self.file_suffix == 'pth':
            file_path = os.path.join(file_dir, '%s.pth' % file_name)
            data = torch.load(file_path)

        return data, label


if __name__ == '__main__':
    # path = r'/workspace/datasets/G-PCD-fixed_64_250'
    # root_file = r'/workspace/datasets/G-PCD-fixed_64_250/subj_desktop_dsis_joint.csv'
    # dataset = Pairwise_Patch_Point_Cloud_Tune_GPCD_Input(path, root_file, 'pth')
    # d, l = dataset[0]
    # print(d.shape)
    # print(l)
    root_path = '/workspace/Projs/SSL_Multitasking/Datasets/Boot_file'
    dataset = PairwsiePatchPointCloudTunePLD(root_path, 'score_train.txt')

    print(dataset.__len__())
    x, y = dataset[1]
    print(x.shape, y)
