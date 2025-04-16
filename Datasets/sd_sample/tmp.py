# -*- coding: utf-8 -*-
"""
@author lizheng
@date  9:36
@packageName
@className tmp
@software PyCharm
@version 1.0.0
@describe TODO
"""

import numpy as np

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

if __name__ == '__main__':
    # 极径
    r = 5

    phi = 0
    n = 16

    # Shoot every 45 degrees and calculate the position of the camera
    for i in range(n):
        angle = i * (360 / n)
        theta = -180 + angle  # calculate the current vertical Angle
        theta_rad = np.radians(theta)

        # spherical coordinates 2 cartesian coordinates
        x = r * np.sin(theta_rad) * np.cos(np.radians(phi))
        y = r * np.sin(theta_rad) * np.sin(np.radians(phi))
        z = r * np.cos(theta_rad)

        transposition_matrix = euler_to_rotation_matrix([0, 90, 0])
        print(i, angle, x, y, z)
        # print(f"At {theta} degrees - Camera position (x, y, z): ({x}, {y}, {z})")
