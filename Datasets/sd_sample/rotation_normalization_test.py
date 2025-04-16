# -*- coding: utf-8 -*-
"""
@author lizheng
@date  9:11
@packageName
@className rotation_normalization_test
@software PyCharm
@version 1.0.0
@describe TODO
"""

import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as R


def normalize_vector(vec):
    return vec / np.linalg.norm(vec)


def rotation_matrix_from_vectors(vectorBefore, vectorAfter):
    # 将向量转为单位向量
    va = normalize_vector(vectorBefore)
    vb = normalize_vector(vectorAfter)

    # 计算旋转轴
    vs = np.cross(vb, va)
    v = normalize_vector(vs)

    # 计算旋转角的余弦值
    ca = np.dot(vb, va)

    # 计算缩放
    scale = 1 - ca

    # 计算旋转矩阵
    rm = np.eye(3)
    rm[0, 0] = v[0] * v[0] + ca
    rm[1, 1] = v[1] * v[1] + ca
    rm[2, 2] = v[2] * v[2] + ca

    vt = scale * v
    rm[0, 1] = vt[0] - vs[2]
    rm[0, 2] = vt[2] + vs[1]
    rm[1, 0] = vt[0] + vs[2]
    rm[1, 2] = vt[1] - vs[0]
    rm[2, 0] = vt[2] - vs[1]
    rm[2, 1] = vt[1] + vs[0]

    return rm


if __name__ == '__main__':
    # front_path = r'E:\homegate\R_Quality_Assessment\tmp\tmp_front.ply'
    # back_path = r'E:\homegate\R_Quality_Assessment\tmp\tmp_bak.ply'
    #
    # rotation_matrix = np.array([[1, 0, 0],
    #                             [0, -1, 0],
    #                             [0, 0, -1]])
    #
    # front_pcd = o3d.io.read_point_cloud(front_path)
    # back_pcd = o3d.io.read_point_cloud(back_path)
    #
    # front_pts = np.asarray(front_pcd.points)
    # back_pts = np.asarray(back_pcd.points)
    #
    # # camera radius center by camera center
    # radius = 5
    # x, y, z = 0, 0, 1
    # front_pts += radius * np.array([x, y, z])
    # back_pts += radius * np.array([x, y, z])
    # print(front_pts)
    # print(back_pts)
    #
    # # rotate back_pts
    # rotated_back_pts = np.dot(back_pts, rotation_matrix)
    # rotated_back_pcd = o3d.geometry.PointCloud()
    # rotated_back_pcd.points = o3d.utility.Vector3dVector(rotated_back_pts)
    # print(rotated_back_pts)
    # save_path = r'E:\homegate\R_Quality_Assessment\tmp\tmp_rotated_back.ply'
    # o3d.io.write_point_cloud(save_path, rotated_back_pcd)
    #
    # # write back
    # front_pcd.points = o3d.utility.Vector3dVector(front_pts)
    # back_pcd.points = o3d.utility.Vector3dVector(back_pts)
    # o3d.io.write_point_cloud(front_path, front_pcd)
    # o3d.io.write_point_cloud(back_path, back_pcd)

    r = 5
    n = 16
    phi = 90


    def euler_to_rotation_matrix(euler_angles):
        # 将欧拉角转换为弧度
        euler_angles_rad = np.radians(euler_angles)

        # 分别计算绕 x、y、z 轴的旋转矩阵
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(euler_angles_rad[0]), -np.sin(euler_angles_rad[0])],
                        [0, np.sin(euler_angles_rad[0]), np.cos(euler_angles_rad[0])]])

        R_y = np.array([[np.cos(euler_angles_rad[1]), 0, np.sin(euler_angles_rad[1])],
                        [0, 1, 0],
                        [-np.sin(euler_angles_rad[1]), 0, np.cos(euler_angles_rad[1])]])

        R_z = np.array([[np.cos(euler_angles_rad[2]), -np.sin(euler_angles_rad[2]), 0],
                        [np.sin(euler_angles_rad[2]), np.cos(euler_angles_rad[2]), 0],
                        [0, 0, 1]])

        # 计算最终的旋转矩阵
        rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))

        return rotation_matrix


    def angle_between_vectors(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cos_theta = dot_product / (norm_a * norm_b)
        angle = np.arccos(cos_theta)
        return np.degrees(angle)


    # Shoot every 45 degrees and calculate the position of the camera
    angle = 0 * (360 / n)
    theta = -180 + angle  # calculate the current vertical Angle
    x1 = 0
    y1 = r * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    z1 = r * np.cos(np.radians(theta))
    vector1 = np.array([x1, y1, z1])
    for i in range(n):
        angle = i * (360 / n)
        theta = -180 + angle  # calculate the current vertical Angle
        theta_rad = np.radians(theta)

        # spherical coordinates 2 cartesian coordinates
        x = r * np.sin(theta_rad) * np.cos(np.radians(phi))
        y = r * np.sin(theta_rad) * np.sin(np.radians(phi))
        z = r * np.cos(theta_rad)

        angle = angle_between_vectors(vector1[1:], [y, z])
        euler_angles = [angle, 0, 0]

        # rotation_matrix = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
        rotation_matrix = euler_to_rotation_matrix(euler_angles)
        if theta > 0:
            # transposition_matrix = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
            transposition_matrix = euler_to_rotation_matrix([180, 0, 0])
            rotation_matrix = np.dot(rotation_matrix, transposition_matrix)
            # transposition_matrix = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
            transposition_matrix = euler_to_rotation_matrix([0, 180, 0])
            rotation_matrix = np.dot(rotation_matrix, transposition_matrix)

        load_file = f'E:/homegate/R_Quality_Assessment/tmp/tmp_phi{phi}_theta{theta}.ply'
        save_file = f'E:/homegate/R_Quality_Assessment/tmp/tmp_phi{phi}_theta{theta}_rotated.ply'

        pcd = o3d.io.read_point_cloud(load_file)
        pts = np.asarray(pcd.points)

        projected_pts = np.dot(pts, rotation_matrix) if not np.isnan(rotation_matrix).all() else pts

        # # 相机位移回归，绝对坐标
        projected_pts -= np.array([x, y, z])

        print(angle, theta, [x, y, z])
        print(projected_pts)

        projected_pcd = o3d.geometry.PointCloud()
        projected_pcd.points = o3d.utility.Vector3dVector(projected_pts)
        o3d.io.write_point_cloud(save_file, projected_pcd)
