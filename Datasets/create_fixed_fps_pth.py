# -*- coding: utf-8 -*-
"""
@author lizheng
@date 2023年01月11日 11:30:57
@packageName 
@className create_fixed_fps_pth
@version 1.0.0
@describe
"""

import torch
import os
import numpy as np
import open3d as o3d
from FPS import farthest_point_sample, index_points, query_ball_point
from tqdm import tqdm


def data10_create_pth(path_src, path_dest, patch_num, patch_size):
    distortion_type_list = ["gaussian_noise", "uniform_noise", "impulse_noise", "index_noise",
                            "random_downsample", "gridAverage_downsample", "OctreeCom"]
    if not os.path.exists(path_dest):
        os.mkdir(path_dest)

    proc_bar = tqdm(total=150 * 7 * 11)

    for model_num in range(150):
        for distortion_type in distortion_type_list:
            src_dir_path = os.path.join(path_src, 'raw_model_%d' % model_num)
            dest_dir_path = os.path.join(path_dest, 'raw_model_%d' % model_num)
            if not os.path.exists(dest_dir_path):
                os.mkdir(dest_dir_path) 

            src_dir_path = os.path.join(src_dir_path, distortion_type)
            dest_dir_path = os.path.join(dest_dir_path, distortion_type)
            if not os.path.exists(dest_dir_path):
                os.mkdir(dest_dir_path)

            file_list = os.listdir(src_dir_path)
            for file_name in file_list:
                proc_bar.set_description(f"calculate raw_model_{model_num}")
                proc_bar.update()

                load_file_path = os.path.join(src_dir_path, file_name)
                save_file_path = os.path.join(dest_dir_path, "%s.pth" % file_name[:-4])

                # print('load_file_path:', load_file_path)
                # print('save_file_path:', save_file_path, '\n')

                # 做FPS
                points = np.asarray(o3d.io.read_point_cloud(load_file_path).points).reshape((1, -1, 3))
                points = torch.from_numpy(points).to(torch.float32)  # type: torch.Tensor
                points = points.cuda()
                # print('points.shape', points.shape)

                # 对模型进行采样，取patch
                centroids_index = farthest_point_sample(points, patch_num)  # 每次采样点数
                centroids = index_points(points, centroids_index)  # centroids:[B S C]
                # radius采样
                result = query_ball_point(0.2, patch_size, points, centroids)  # result:[B S nsample]
                B, S, patch_size = result.shape
                data_tensor = torch.zeros((B, S, patch_size, 3), dtype=torch.float32)
                for patch in range(S):  # 0-64
                    patch_index = result[:, patch, :]  # [B nsample]，nsample=patch_size
                    value = index_points(points, patch_index)  # value:[B patch_size C]
                    for batch in range(B):
                        data_tensor[batch][patch] = value[batch]  # result_value:[B S patch_size C],S*patch_size=N
                data_tensor = data_tensor[0].cpu()
                torch.save(data_tensor, save_file_path)
                # print('data_tensor.shape', data_tensor.shape)
                # exit(103)


def g_pcd_create_pth(path_src, path_dest, patch_num, patch_size):
    if not os.path.exists(path_dest):
        os.mkdir(path_dest)

    file_list = os.listdir(path_src)
    for file_name in file_list:
        load_file_path = os.path.join(path_src, file_name)
        save_file_path = os.path.join(path_dest, "%s.pth" % file_name[:-4])

        # 做FPS
        points = np.asarray(o3d.io.read_point_cloud(load_file_path).points).reshape((1, -1, 3))
        points = torch.from_numpy(points).to(torch.float32)  # type: torch.Tensor
        points = points.cuda()

        # 对模型进行采样，取patch
        centroids_index = farthest_point_sample(points, patch_num)  # 每次采样点数
        centroids = index_points(points, centroids_index)  # centroids:[B S C]
        # radius采样
        result = query_ball_point(0.2, patch_size, points, centroids)  # result:[B S nsample]
        B, S, patch_size = result.shape
        data_tensor = torch.zeros((B, S, patch_size, 3), dtype=torch.float32)
        for patch in range(S):  # 0-64
            patch_index = result[:, patch, :]  # [B nsample]，nsample=patch_size
            value = index_points(points, patch_index)  # value:[B patch_size C]
            for batch in range(B):
                data_tensor[batch][patch] = value[batch]  # result_value:[B S patch_size C],S*patch_size=N
        data_tensor = data_tensor[0].cpu()
        torch.save(data_tensor, save_file_path)


if __name__ == '__main__':
    # '''Data 10 create fixed_fps pth'''
    # path_data10 = r'E:\homegate\R_Quality_Assessment\Dataset\Data10_large_scale'
    # patch_num = 64
    # patch_size = 128
    #
    # path_data10_fixed = r'E:\homegate\R_Quality_Assessment\Dataset\Data10_large_scale_fixed_fps_%d_%d' % (
    #     patch_num, patch_size)
    #
    # data10_create_pth(path_data10, path_data10_fixed, patch_num, patch_size)
    #
    # # # visualization
    # # os.chdir(r'E:\homegate\R_Quality_Assessment\SSL_Proj\visualization\patch_3d_view')
    # # pcd = o3d.geometry.PointCloud()
    # # fps_64_256 = torch.load(
    # #     r'E:\homegate\R_Quality_Assessment\Dataset\Data10_fixed_fps_64_256\raw_model_0\gaussian_noise\level5.pth')
    # # fps_64_128 = torch.load(
    # #     r'E:\homegate\R_Quality_Assessment\Dataset\Data10_fixed_fps_64_128\raw_model_0\gaussian_noise\level5.pth')
    # #
    # # fps_64_256 = fps_64_256.view(-1, 3).numpy()
    # # fps_64_128 = fps_64_128.view(-1, 3).numpy()
    # #
    # # pcd.points = o3d.utility.Vector3dVector(fps_64_256)
    # # o3d.io.write_point_cloud('fps_64_256.ply', pcd)
    # #
    # # pcd.points = o3d.utility.Vector3dVector(fps_64_128)
    # # o3d.io.write_point_cloud('fps_64_128.ply', pcd)

    '''G-PCD create fixed_fps pth'''
    g_pcd_src_path = r'E:\tmp\G-PCD\stimuli\D02'
    g_pcd_dst_path = r'E:\tmp\G-PCD-fixed_64_250\D02_fixed_fps'

    patch_num = 64
    patch_size = 250

    g_pcd_create_pth(g_pcd_src_path, g_pcd_dst_path, patch_num, patch_size)
