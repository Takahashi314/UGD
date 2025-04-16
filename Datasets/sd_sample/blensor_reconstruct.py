# -*- coding: utf-8 -*-
"""
@author lizheng
@date  13:51
@packageName
@className blensor_reconstruct
@software PyCharm
@version 1.0.0
@describe TODO
"""

import os
from pathlib import Path

import bpy
import blensor
import numpy as np
from mathutils import *


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


def three_view_sample(file_path, n):
    r = 5

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    bpy.ops.import_mesh.ply(filepath=str(file_path))

    bpy.ops.object.select_by_type(type='CAMERA')
    objects = bpy.context.scene.objects
    object_name = "Camera"

    for obj in objects:
        if obj.name == object_name:
            obj.select = True
            bpy.context.scene.objects.active = obj
        else:
            obj.select = False

    total_pts = []

    phi = 90
    # Initialize Camera Vector
    angle = 0 * (360 / n)
    theta = -180 + angle  # calculate the current vertical Angle
    x1 = 0
    y1 = r * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    z1 = r * np.cos(np.radians(theta))
    vector1 = np.array([x1, y1, z1])

    # Shoot every 45 degrees and calculate the position of the camera
    # X axis reconstruction
    for i in range(n):
        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('NoisyScan'):
                bpy.data.objects.remove(item)
            elif item.type == 'MESH' and item.name.startswith('Scan'):
                bpy.data.objects.remove(item)
        angle = i * (360 / n)
        theta = -180 + angle  # calculate the current vertical Angle
        theta_rad = np.radians(theta)

        # spherical coordinates 2 cartesian coordinates
        x = r * np.sin(theta_rad) * np.cos(np.radians(phi))
        y = r * np.sin(theta_rad) * np.sin(np.radians(phi))
        z = r * np.cos(theta_rad)

        l = np.linalg.norm([x, y, z])
        bpy.data.objects['Camera'].location[0] = x
        bpy.data.objects['Camera'].location[1] = y
        bpy.data.objects['Camera'].location[2] = z
        bpy.data.objects['Camera'].rotation_euler[0] = np.arccos(z / l)
        bpy.data.objects['Camera'].rotation_euler[1] = 0 * np.pi / 180
        if y < 0:
            bpy.data.objects['Camera'].rotation_euler[2] = np.arctan(-x / (y))
        elif y > 0:
            bpy.data.objects['Camera'].rotation_euler[2] = np.arctan(-x / (y)) + np.pi
        elif y == 0:
            bpy.data.objects['Camera'].rotation_euler[2] = -np.arctan(-x / (y + 0.0001))

        bpy.ops.blensor.delete_scans()
        bpy.ops.blensor.scan()

        pts = []
        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('Scan'):
                for sp in item.data.vertices:
                    pts.append([sp.co[0], sp.co[1], sp.co[2]])
        pts = np.array(pts)

        angle = angle_between_vectors(vector1[[1, 2]], [y, z])
        euler_angles = [angle, 0, 0]

        # calculate the rotation matrix
        # rotation_matrix = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
        rotation_matrix = euler_to_rotation_matrix(euler_angles)
        if theta > 0:
            # transposition_matrix = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
            transposition_matrix = euler_to_rotation_matrix([180, 0, 0])
            rotation_matrix = np.dot(rotation_matrix, transposition_matrix)
            # transposition_matrix = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
            transposition_matrix = euler_to_rotation_matrix([0, 180, 0])
            rotation_matrix = np.dot(rotation_matrix, transposition_matrix)

        projected_pts = np.dot(pts, rotation_matrix)
        # point cloud shift to absolute coordinate based on the camera position  ||| Important !!!!
        projected_pts -= np.array([x, y, z])
        total_pts.append(projected_pts)

    phi = 0
    # Initialize Camera Vector
    angle = 0 * (360 / n)
    theta = -180 + angle  # calculate the current vertical Angle
    x1 = r * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    y1 = r * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    z1 = r * np.cos(np.radians(theta))
    vector1 = np.array([x1, y1, z1])

    # Shoot every 45 degrees and calculate the position of the camera
    # X axis reconstruction
    for i in range(n):
        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('NoisyScan'):
                bpy.data.objects.remove(item)
            elif item.type == 'MESH' and item.name.startswith('Scan'):
                bpy.data.objects.remove(item)
        angle = i * (360 / n)
        theta = -180 + angle  # calculate the current vertical Angle
        theta_rad = np.radians(theta)

        # spherical coordinates 2 cartesian coordinates
        x = r * np.sin(theta_rad) * np.cos(np.radians(phi))
        y = r * np.sin(theta_rad) * np.sin(np.radians(phi))
        z = r * np.cos(theta_rad)

        l = np.linalg.norm([x, y, z])
        bpy.data.objects['Camera'].location[0] = x
        bpy.data.objects['Camera'].location[1] = y
        bpy.data.objects['Camera'].location[2] = z
        bpy.data.objects['Camera'].rotation_euler[0] = np.arccos(z / l)
        bpy.data.objects['Camera'].rotation_euler[1] = 0 * np.pi / 180
        if y < 0:
            bpy.data.objects['Camera'].rotation_euler[2] = np.arctan(-x / (y))
        elif y > 0:
            bpy.data.objects['Camera'].rotation_euler[2] = np.arctan(-x / (y)) + np.pi
        elif y == 0:
            bpy.data.objects['Camera'].rotation_euler[2] = -np.arctan(-x / (y + 0.0001))

        bpy.ops.blensor.delete_scans()
        bpy.ops.blensor.scan()

        pts = []
        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('Scan'):
                for sp in item.data.vertices:
                    pts.append([sp.co[0], sp.co[1], sp.co[2]])
        pts = np.array(pts)

        angle = angle_between_vectors(vector1[[0, 2]], [x, z])
        euler_angles = [angle, 0, 0]

        # calculate the rotation matrix
        rotation_matrix = euler_to_rotation_matrix(euler_angles)
        if theta == -180:
            transposition_matrix = euler_to_rotation_matrix([0, 0, 90])
            rotation_matrix = np.dot(rotation_matrix, transposition_matrix)
        elif theta == 0:
            transposition_matrix = euler_to_rotation_matrix([0, 0, 90])
            rotation_matrix = np.dot(rotation_matrix, transposition_matrix)
        elif theta > 0:
            transposition_matrix = euler_to_rotation_matrix([0, 0, 180])
            rotation_matrix = np.dot(rotation_matrix, transposition_matrix)

        projected_pts = np.dot(pts, rotation_matrix)
        # point cloud shift to absolute coordinate based on the camera position  ||| Important !!!!
        projected_pts -= np.array([y, x, z])

        rotation_matrix = euler_to_rotation_matrix([0, 0, -90])
        projected_pts = np.dot(projected_pts, rotation_matrix)
        total_pts.append(projected_pts)

    # Shoot every 45 degrees and calculate the position of the camera
    # X axis reconstruction
    phi = 90
    angle = 0 * (360 / n)
    theta = 180 + angle  # calculate the current vertical Angle
    x1 = r * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    y1 = r * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    z1 = r * np.cos(np.radians(theta))
    vector1 = np.array([x1, y1, z1])

    # Shoot every 45 degrees and calculate the position of the camera
    # X axis reconstruction
    for i in range(n):
        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('NoisyScan'):
                bpy.data.objects.remove(item)
            elif item.type == 'MESH' and item.name.startswith('Scan'):
                bpy.data.objects.remove(item)
        angle = i * (360 / n)
        theta = 0 + angle - 90
        phi_rad = np.radians(phi)

        # spherical coordinates 2 cartesian coordinates
        x = r * np.sin(phi_rad) * np.cos(np.radians(theta))
        y = r * np.sin(phi_rad) * np.sin(np.radians(theta))
        z = r * np.cos(phi_rad)

        l = np.linalg.norm([x, y, z])
        bpy.data.objects['Camera'].location[0] = x
        bpy.data.objects['Camera'].location[1] = y
        bpy.data.objects['Camera'].location[2] = z
        bpy.data.objects['Camera'].rotation_euler[0] = np.arccos(z / l)
        bpy.data.objects['Camera'].rotation_euler[1] = 0 * np.pi / 180
        if y < 0:
            bpy.data.objects['Camera'].rotation_euler[2] = np.arctan(-x / (y))
        elif y > 0:
            bpy.data.objects['Camera'].rotation_euler[2] = np.arctan(-x / (y)) + np.pi
        elif y == 0:
            bpy.data.objects['Camera'].rotation_euler[2] = -np.arctan(-x / (y + 0.0001))

        bpy.ops.blensor.delete_scans()
        bpy.ops.blensor.scan()

        pts = []
        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('Scan'):
                for sp in item.data.vertices:
                    pts.append([sp.co[0], sp.co[1], sp.co[2]])
        pts = np.array(pts)

        angle = angle_between_vectors(vector1[[0, 1]], [x, y])
        euler_angles = [0, angle, 0]

        # calculate the rotation matrix
        rotation_matrix = euler_to_rotation_matrix(euler_angles)
        if theta > 90:
            transposition_matrix = euler_to_rotation_matrix([0, 180, 0])
            rotation_matrix = np.dot(rotation_matrix, transposition_matrix)

        projected_pts = np.dot(pts, rotation_matrix)
        if theta > 90:
            tmp_pts = projected_pts + np.array([-x, z, -y])
            res_degree = theta - 90
            rotation_matrix = euler_to_rotation_matrix([0, 180, 0])
            rotation_matrix = np.dot(rotation_matrix, euler_to_rotation_matrix([0, -2 * res_degree, 0]))
            projected_pts = np.dot(tmp_pts, rotation_matrix) - np.array([-x, z, -y])

        # point cloud shift to absolute coordinate based on the camera position  ||| Important !!!!
        if theta > 90:
            projected_pts += np.array([-x, z, -y])
        else:
            projected_pts += np.array([-x, z, y])

        rotation_matrix = euler_to_rotation_matrix([-90, 0, 0])
        rotation_matrix = np.dot(rotation_matrix, euler_to_rotation_matrix([0, 180, 0]))
        projected_pts = np.dot(projected_pts, rotation_matrix)
        total_pts.append(projected_pts)

    total_pts = np.concatenate(total_pts, axis=0)
    return total_pts


def print_progress_bar(iteration, total, length=50):
    """
    显示红色进度条，包含已完成个数和总个数
    :param iteration: 当前进度
    :param total: 总进度
    :param length: 进度条的长度
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)

    # 使用ANSI转义码输出红色进度条，并显示已完成和总个数
    print(f'\n\r\033[91mProgress: |{bar}| {percent}% Complete ({iteration}/{total})\033[0m', end='\n\n')

    if iteration == total:
        print()  # 在进度条完成后换行


def main():
    root_dir = r'E:\homegate\R_Quality_Assessment\Dataset\or150PStanford_mesh'
    dst_dir = r'E:\homegate\R_Quality_Assessment\Dataset\or150PStanford_data5'
    files = list(Path(root_dir).glob("*.ply"))
    n_list = [16, 12, 8, 6, 3]
    bar_total = len(files) * len(n_list)
    cnt = 0

    for file in files:
        name = file.stem
        noise_dir = Path(dst_dir) / name / 'sd'
        noise_dir.mkdir(parents=True, exist_ok=True)
        for i, n in enumerate(n_list):
            pts = three_view_sample(str(file), n)
            save_file = noise_dir / f"level{i + 1}.txt"
            np.savetxt(save_file, pts)
            cnt += 1
            print_progress_bar(cnt, bar_total)
        break


main()
