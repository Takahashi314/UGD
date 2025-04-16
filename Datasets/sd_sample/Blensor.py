# -*- coding: utf-8 -*-
"""
@author lizheng
@date  15:04
@packageName
@className Blensor
@software PyCharm
@version 1.0.0
@describe TODO
"""

import numpy as np
from pathlib import Path
from mathutils import *
import bpy
import blensor
import os
import random
import math

root_dir = r'E:\homegate\R_Quality_Assessment\Dataset\or150PStanford_mesh'
dst_dir = r'E:\homegate\R_Quality_Assessment\Dataset\or150PStanford_data5'
files = list(Path(root_dir).glob("*.ply"))

for i, file_path in enumerate(files):
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

    scan_angles = [25, 35, 45, 55, 65]

    for angle_idx, angle in enumerate(scan_angles):
        bpy.data.objects["Camera"].tof_lens_angle_w = angle
        bpy.data.objects["Camera"].tof_lens_angle_h = angle

        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('NoisyScan'):
                bpy.data.objects.remove(item)
            elif item.type == 'MESH' and item.name.startswith('Scan'):
                bpy.data.objects.remove(item)

        bpy.ops.blensor.delete_scans()
        bpy.ops.blensor.scan()

        pc_dst_dir = Path(dst_dir) / file_path.stem / 'sd'
        pc_dst_dir.mkdir(parents=True, exist_ok=True)
        r = 5
        j = 0
        while j < 14:
            if j == 0:
                x = r
                y = 0
                z = 0
            elif j == 1:
                x = -r
                y = 0
                z = 0
            elif j == 2:
                x = 0
                y = r
                z = 0
            elif j == 3:
                x = 0
                y = -r
                z = 0
            elif j == 4:
                x = 0
                y = 0
                z = r
            elif j == 5:
                x = 0
                y = 0
                z = -r
            elif j == 6:
                x = math.sqrt(r)
                y = math.sqrt(r)
                z = math.sqrt(r)
            elif j == 7:
                x = -math.sqrt(r)
                y = math.sqrt(r)
                z = math.sqrt(r)
            elif j == 8:
                x = math.sqrt(r)
                y = -math.sqrt(r)
                z = math.sqrt(r)
            elif j == 9:
                x = -math.sqrt(r)
                y = -math.sqrt(r)
                z = math.sqrt(r)
            elif j == 10:
                x = math.sqrt(r)
                y = math.sqrt(r)
                z = -math.sqrt(r)
            elif j == 11:
                x = -math.sqrt(r)
                y = math.sqrt(r)
                z = -math.sqrt(r)
            elif j == 12:
                x = math.sqrt(r)
                y = -math.sqrt(r)
                z = -math.sqrt(r)
            elif j == 13:
                x = -math.sqrt(r)
                y = -math.sqrt(r)
                z = -math.sqrt(r)
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

            filename = pc_dst_dir / f'level{angle_idx + 1}.txt'
            # os.makedirs(dir_path)
            f = open(filename, "w")
            i = 0  # Store the number of points in the point cloud
            for item in bpy.data.objects:
                if item.type == 'MESH' and item.name.startswith('Scan'):
                    # print('write once')
                    for sp in item.data.vertices:
                        str = '%#5.3f\t%#5.3f\t%#5.3f \n' % (sp.co[0], sp.co[1], sp.co[2])
                        i = i + 1
                        f.write(str)
            f.close()
            j += 1
            bpy.ops.blensor.delete_scans()
            bpy.ops.blensor.scan()


for k in range(150):
    file_path = r""
    file_path = os.path.join(file_path + '\\raw_model_%d.ply' % k)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    bpy.ops.import_mesh.ply(filepath=file_path)
    bpy.ops.object.select_by_type(type='CAMERA')
    objects = bpy.context.scene.objects
    object_name = "Camera"
    for obj in objects:
        if obj.name == object_name:
            obj.select = True
            bpy.context.scene.objects.active = obj
        else:
            obj.select = False
    for n in range(20):
        l = [25, 27, 30, 33, 35, 37, 40, 43, 45, 47, 50, 53, 55, 57, 60, 63, 65, 67, 70, 73]
        bpy.data.objects["Camera"].tof_lens_angle_w = l[n]
        bpy.data.objects["Camera"].tof_lens_angle_h = l[n]

        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('NoisyScan'):
                bpy.data.objects.remove(item)
            elif item.type == 'MESH' and item.name.startswith('Scan'):
                bpy.data.objects.remove(item)

        bpy.ops.blensor.delete_scans()
        bpy.ops.blensor.scan()

        """location"""
        dir_path = r"D:\PCQA\data\Data20\\raw_model_%d\\tof" % k
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        r = 5
        j = 0
        while j < 14:
            if j == 0:
                x = r
                y = 0
                z = 0
            elif j == 1:
                x = -r
                y = 0
                z = 0
            elif j == 2:
                x = 0
                y = r
                z = 0
            elif j == 3:
                x = 0
                y = -r
                z = 0
            elif j == 4:
                x = 0
                y = 0
                z = r
            elif j == 5:
                x = 0
                y = 0
                z = -r
            elif j == 6:
                x = math.sqrt(r)
                y = math.sqrt(r)
                z = math.sqrt(r)
            elif j == 7:
                x = -math.sqrt(r)
                y = math.sqrt(r)
                z = math.sqrt(r)
            elif j == 8:
                x = math.sqrt(r)
                y = -math.sqrt(r)
                z = math.sqrt(r)
            elif j == 9:
                x = -math.sqrt(r)
                y = -math.sqrt(r)
                z = math.sqrt(r)
            elif j == 10:
                x = math.sqrt(r)
                y = math.sqrt(r)
                z = -math.sqrt(r)
            elif j == 11:
                x = -math.sqrt(r)
                y = math.sqrt(r)
                z = -math.sqrt(r)
            elif j == 12:
                x = math.sqrt(r)
                y = -math.sqrt(r)
                z = -math.sqrt(r)
            elif j == 13:
                x = -math.sqrt(r)
                y = -math.sqrt(r)
                z = -math.sqrt(r)
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

            filename = dir_path + '\\' + 'level%d.txt' % n
            # os.makedirs(dir_path)
            f = open(filename, "w")
            i = 0  # Store the number of points in the point cloud
            for item in bpy.data.objects:
                if item.type == 'MESH' and item.name.startswith('Scan'):
                    # print('write once')
                    for sp in item.data.vertices:
                        str = '%#5.3f\t%#5.3f\t%#5.3f \n' % (sp.co[0], sp.co[1], sp.co[2])
                        i = i + 1
                        f.write(str)
            f.close()
            j += 1
            bpy.ops.blensor.delete_scans()
            bpy.ops.blensor.scan()
#
