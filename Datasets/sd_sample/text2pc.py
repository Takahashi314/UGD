# -*- coding: utf-8 -*-
"""
@author lizheng
@date  16:20
@packageName
@className text2pc
@software PyCharm
@version 1.0.0
@describe TODO
"""

from pathlib import Path

import numpy as np
import open3d as o3d

# root_dir = r'E:\homegate\R_Quality_Assessment\Dataset\or150PStanford_data5'
root_dir = r'E:\homegate\R_Quality_Assessment\tmp'

if __name__ == '__main__':
    text_paths = Path(root_dir).glob('**/*.txt')
    for text_path in text_paths:
        text = text_path.read_text()
        pts = np.asarray([[float(item) for item in line.split('\t')] for line in text.strip('\n').split('\n')])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(str(text_path).replace('.txt', '.ply'), pcd)
        # exit()
