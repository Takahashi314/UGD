# -*- coding: utf-8 -*-
"""
@author lizheng
@date 2022年12月01日 00:13:17
@packageName 
@className distortion
@version 1.0.0
@describe Done
"""

import numpy as np
import open3d as o3d
import os
import datetime
from plyfile import PlyData, PlyElement

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def add_gauss_noise(pc, mean=0, std=0.):
    """
    :param pc: PointCloud type: numpy.ndarray
    :param mean: 0
    :param std: noise_level*average_edge_l
    :return: PointCloud with noise type: numpy.ndarray
    """
    w, h = pc.shape
    noise = np.random.normal(mean, std, (w, h))
    out = pc + noise
    return out


def add_uniform_noise(pc, le):
    """
    :param pc: PointCloud type: numpy.ndarray
    :param le: average_edge_l*noise_level
    :return: PointCloud with noise type: numpy.ndarray
    """
    w, h = pc.shape
    a = le * 3
    b = -a
    noise = a + (b - a) * np.random.rand(w * h).reshape((w, h))
    out = pc + noise
    return out


def add_impulse_noise(pc, le):
    """
    :param pc: PointCloud type: numpy.ndarray
    :param le: average_edge_l*noise_level
    :return: PointCloud with noise type: numpy.ndarray
    """
    w, h = pc.shape
    a = le * 3
    b = -a
    p = 0.2  # 噪声密度
    x = np.random.rand(w * h).reshape((w, h))
    noise = np.zeros_like(x)
    noise[x < p / 2] = a
    noise[(x > p / 2) * (x < p)] = b
    out = pc + noise
    return out


def add_exponent_noise(pc, le):
    """
    :param pc: PointCloud type: numpy.ndarray
    :param le: average_edge_l*noise_level
    :return: PointCloud with noise type: numpy.ndarray
    """
    w, h = pc.shape
    noise = np.random.exponential(le, (w, h))
    out = pc + noise
    return out


def pc_random_downsample(pc, percentage):
    """
    :param pc: PointCloud type: numpy.ndarray
    :param percentage: level
    :return: PointCloud with noise type: numpy.ndarray
    """
    percentage = 1 - percentage
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd_new = pcd.random_down_sample(percentage)
    out = np.asarray(pcd_new.points)
    return out


def pc_grid_downsample(pc, grid_step):
    """
    :param pc: PointCloud type: numpy.ndarray
    :param grid_step:  average_edge_l*level
    :return: PointCloud with noise type: numpy.ndarray
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd_new = pcd.voxel_down_sample(voxel_size=grid_step)
    out = np.asarray(pcd_new.points)
    return out


def pc_octree_compress(pc, level, is_print_info):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    temp_path = os.path.join(FILE_PATH, 'temp')
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    dt = datetime.datetime.today()
    temp_file_path = '%s_%s_%s_%s_%s_%s_random_%f.ply' % (
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, np.random.random())
    temp_file_path = os.path.join(temp_path, temp_file_path)
    vertex = np.array([tuple(pc[x, :]) for x in range(pc.shape[0])], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(temp_file_path)

    exe_path = os.path.join(FILE_PATH, 'OctreeCom.exe')
    cmd = "%s %s %s %f" % (exe_path, temp_file_path, temp_file_path, level)
    with os.popen(cmd) as f:
        f_str = f.read()
    if is_print_info:
        print(f_str)
    pcd = o3d.io.read_point_cloud(temp_file_path)
    pc_new = np.asarray(pcd.points)

    os.remove(temp_file_path)
    return pc_new


if __name__ == '__main__':
    test_file = r'E:\homegate\R_Quality_Assessment\Dataset\data5\raw_model_0\raw_model_0.ply'
    save_file = r'C:\Users\lizheng\Desktop\test\python_noise.ply'

    # points = np.asarray(o3d.io.read_point_cloud(test_file).points)
    # pc_octree_compress(points, 0.01, False)
