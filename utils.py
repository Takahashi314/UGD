# -*- coding: utf-8 -*-
"""
@author lizheng
@date  21:15
@packageName
@className utils
@software PyCharm
@version 1.0.0
@describe TODO
"""

import time
import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds


def run_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # running_time 取小数点后两位
        running_time = end_time - start_time
        hours, minutes, seconds = format_time(running_time)
        print(
            '\033[33m%s run time: %d hours, %d minutes, %.2f seconds\033[0m'
            % (func.__name__, hours, minutes, seconds))
        return result

    return wrapper


def cal_mean_points_edge(pc_path):
    points = np.asarray(o3d.io.read_point_cloud(pc_path).points)
    kdtree = cKDTree(points)
    distances, indices = kdtree.query(points, k=2)
    dist_list = []
    for i, (point, distance) in enumerate(zip(points, distances)):
        nearest_neighbor_index = indices[i, 1]  # 第一个最近邻点的索引
        nearest_neighbor_distance = distance[1]  # 第一个最近邻点的距离
        dist_list.append(nearest_neighbor_distance)
    return np.mean(dist_list)


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device).to(mat_a.dtype)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


if __name__ == '__main__':
    ...
