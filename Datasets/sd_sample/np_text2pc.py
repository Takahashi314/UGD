# -*- coding: utf-8 -*-
"""
@author lizheng
@date  14:12
@packageName
@className np_text2pc
@software PyCharm
@version 1.0.0
@describe TODO
"""

import os
from pathlib import Path

import numpy as np
import open3d as o3d

if __name__ == '__main__':
    root_path = r'E:\homegate\tmp\or150PStanford_data5'
    root_path = Path(root_path)
    txt_files = list(root_path.rglob('*.txt'))

    for txt in txt_files:
        pcd_path = str(txt).replace('.txt', '.ply')
        pts = np.loadtxt(txt)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(pcd_path, pcd)
        os.remove(str(txt))
