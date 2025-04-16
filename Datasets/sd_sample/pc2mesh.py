# -*- coding: utf-8 -*-
"""
@author lizheng
@date  15:06
@packageName
@className pc2mesh
@software PyCharm
@version 1.0.0
@describe TODO
"""

import os
from pathlib import Path

import open3d as o3d
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    root_dir = r'E:\homegate\R_Quality_Assessment\Dataset\or150PStanford'
    dst_dir = r'E:\homegate\R_Quality_Assessment\Dataset\or150PStanford_mesh'
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    files = Path(root_dir).glob('*.ply')

    for file in tqdm(list(files)):
        # print(file)
        pcd = o3d.io.read_point_cloud(str(file))
        pts = np.asarray(pcd.points)

        # pts smallest point shifts to the origin and scale normalization
        min_point = np.min(pts, axis=0)
        pts -= min_point
        scale = np.max(pts)
        pts /= scale
        # shift point cloud center to the origin
        pts -= np.mean(pts, axis=0)

        pcd.points = o3d.utility.Vector3dVector(pts)

        # calculate normal
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # BPA重建
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

        # # Alpha shapes
        # alpha = 0.03
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        # # 创建泊松网格对象
        # poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
        # # 对泊松网格进行平滑处理
        # mesh = poisson_mesh.filter_smooth_taubin(number_of_iterations=5)

        save_path = os.path.join(dst_dir, file.stem + '.ply')
        o3d.io.write_triangle_mesh(save_path, mesh)
        exit()
