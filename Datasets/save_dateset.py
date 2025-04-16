# -*- coding: utf-8 -*-
"""
@author lizheng
@date  16:43
@packageName
@className save_dateset
@software PyCharm
@version 1.0.0
@describe TODO
"""

import os
import glob
import open3d as o3d
import numpy as np

if __name__ == '__main__':
    raw_dataset_path = r'/workspace/datasets/or150PStanford'
    dis_dataset_path = r'/workspace/datasets/or150PStanford_data5'

    new_or_path_list = [os.path.join(raw_dataset_path, item) for item in os.listdir(raw_dataset_path) if
                        not item.startswith('raw_model')]

    d = {os.path.basename(item)[:-4]: {'or_path': item, 'dis_path_list': None} for item in new_or_path_list}
    for key in d.keys():
        dis_folder = os.path.join(dis_dataset_path, key)
        ply_files = glob.glob(dis_folder + '/**/*.ply')
        ply_files.append(os.path.join(dis_folder, f'{key}.ply'))
        d[key]['dis_path_list'] = ply_files

    for key, value in d.items():
        print(key)
        or_pcd = o3d.io.read_point_cloud(value['or_path'])
        or_points = np.asarray(or_pcd.points)
        max_p = np.max(or_points)
        print(or_points.shape, max_p)

        or_points /= max_p
        or_pcd.points = o3d.utility.Vector3dVector(or_points)
        o3d.io.write_point_cloud(value['or_path'], or_pcd)

        for ply_path in value['dis_path_list']:
            pcd = o3d.io.read_point_cloud(ply_path)
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) / max_p)
            o3d.io.write_point_cloud(ply_path, pcd)

        # break
