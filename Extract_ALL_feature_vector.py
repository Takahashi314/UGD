# -*- coding: utf-8 -*-
"""
@author lizheng
@date  14:59
@packageName
@className Extract_ALL_feature_vector
@software PyCharm
@version 1.0.0
@describe TODO
"""
import os
import re
import argparse

import torch
import pandas as pd
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils import run_time
from Datasets.FPS import fps_sampling_func
# from SSL_Multitasking.Model.dgcnn_model import DGCNN
from Model.pct import Point_Transformer
from Model.subtask import Weight_Net

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
FILE_PATH = os.path.dirname(os.path.abspath(__file__))


class FeatureDatasets(Dataset):
    def __init__(self, raw_dataset_path, dis_dataset_path, feature_dataset_path
                 , patch_num=64, patch_size=250, level_len=5,
                 multiple_factor=1):
        self._root_path = ROOT_PATH
        self._patch_num = patch_num
        self._patch_size = patch_size
        self._level_len = level_len
        self._multiple_factor = multiple_factor
        self._raw_dataset_path = raw_dataset_path
        self._dis_dataset_path = dis_dataset_path
        self._feature_dataset_path = feature_dataset_path

        self._dis_dict = {
            'GN': 'noise1',
            'UN': 'noise2',
            'IN': 'noise3',
            'EN': 'noise4',
            'SD': 'sd',
        }

        train_csv_path, test_csv_path = self.create_index_file()
        self._train_index = pd.read_csv(train_csv_path)
        self._test_index = pd.read_csv(test_csv_path)
        self._index = pd.concat([self._train_index, self._test_index], ignore_index=True)
        self._fixed_fps_centroids_dict = {}
        # 两个dataframe合并

    @staticmethod
    def _get_level(x):
        if re.findall(r'level\d', x):
            return int(re.findall(r'level\d', x)[0][-1])
        else:
            return 0

    def create_index_file(self):
        train_csv = pd.read_csv(os.path.join(FILE_PATH, 'Datasets/random_split_dir', 'train_split.csv'))
        test_csv = pd.read_csv(os.path.join(FILE_PATH, 'Datasets/random_split_dir', 'test_split.csv'))
        train_output_index = pd.DataFrame(columns=['ply_file_path', 'feature_vector_path', 'type'])
        test_output_index = pd.DataFrame(columns=['ply_file_path', 'feature_vector_path', 'type'])
        input_csvs = [train_csv, test_csv]
        output_csvs = [train_output_index, test_output_index]

        pbar = tqdm(total=train_csv.__len__() + test_csv.__len__())
        for step, input_csv in enumerate(input_csvs):
            model_list = [i[:-4] for i in input_csv['filename']]
            for model in model_list:
                pbar.update()
                pbar.set_description('Create index file')
                input_stem_dir = os.path.join(self._dis_dataset_path, model)
                output_stem_dir = os.path.join(self._feature_dataset_path, model)
                ground_truth_path = os.path.join(self._dis_dataset_path, model, '%s.ply' % model)

                input_ply_path = ground_truth_path
                output_feature_path = os.path.join(output_stem_dir, '%s_GT.npy' % model)
                output_csvs[step].loc[output_csvs[step].__len__(), :] = [input_ply_path, output_feature_path, 'GT']

                for dis in self._dis_dict:
                    for level in range(1, self._level_len + 1):
                        input_ply_path = os.path.join(input_stem_dir, self._dis_dict[dis], 'level%d.ply' % level)
                        output_feature_path = os.path.join(output_stem_dir, self._dis_dict[dis], 'level%d.npy' % level)
                        output_csvs[step].loc[output_csvs[step].__len__(), :] = \
                            [input_ply_path, output_feature_path, dis]

        train_output_index['reference_model'] = train_output_index['ply_file_path'].apply(
            lambda x: x.replace(f"{self._dis_dataset_path}/", '').split('/')[0])
        test_output_index['reference_model'] = test_output_index['ply_file_path'].apply(
            lambda x: x.replace(f"{self._dis_dataset_path}/", '').split('/')[0])
        train_output_index['level'] = train_output_index['ply_file_path'].apply(self._get_level)
        test_output_index['level'] = test_output_index['ply_file_path'].apply(self._get_level)

        Path('./boot_file').mkdir(parents=True, exist_ok=True)
        train_csv_path = './boot_file/feature_index_train.csv'
        test_csv_path = './boot_file/feature_index_test.csv'
        train_output_index.to_csv(train_csv_path, index=False)
        test_output_index.to_csv(test_csv_path, index=False)
        return train_csv_path, test_csv_path

    def __len__(self):
        return self._index.__len__()

    def __getitem__(self, item):
        ply_file_path = self._index.loc[item, 'ply_file_path']
        feature_dist_path = self._index.loc[item, 'feature_vector_path']

        data = torch.zeros((self._patch_num * self._multiple_factor, self._patch_size, 3))
        points = torch.from_numpy(np.asarray(o3d.io.read_point_cloud(ply_file_path).points))

        reference_model = self._index.loc[item, 'reference_model']

        if points.shape[0] < self._patch_size:
            return [0], [0]
        for i in range(self._multiple_factor):
            centroids = self._fixed_fps_centroids_dict[reference_model][i, :, :, :]
            # print(centroids.shape)
            points_with_patch, _ = fps_sampling_func(points, self._patch_num, self._patch_size, centroids)
            data[i * 64: (i + 1) * 64, :, :] = points_with_patch
        return data, feature_dist_path

    def save_centroids_dict(self):
        print('self._fixed_fps_centroids_dict.__len__()', self._fixed_fps_centroids_dict.__len__())
        folder_dir = os.path.join(self._feature_dataset_path, 'fixed_centroids')
        if not os.path.exists(folder_dir):
            Path(folder_dir).mkdir(parents=True, exist_ok=True)
            # os.mkdir(folder_dir)
        for key in self._fixed_fps_centroids_dict:
            data = self._fixed_fps_centroids_dict[key].numpy()
            save_path = os.path.join(folder_dir, 'centroids_%s.npy' % key)
            np.save(save_path, data)

    def reference_fps_centroids_sampling(self):
        reference_idx = self._index[self._index['level'] == 0]
        reference_idx.index = range(reference_idx.__len__())
        pbar = tqdm(total=reference_idx.__len__() * self._multiple_factor)
        pbar.set_description('Reference fps centroids sampling')
        for i in range(reference_idx.__len__()):
            centroids_group = torch.zeros(self._multiple_factor, 1, self._patch_num, 3)
            path = reference_idx.loc[i, 'ply_file_path']
            reference_model = reference_idx.loc[i, 'reference_model']
            points = torch.from_numpy(np.asarray(o3d.io.read_point_cloud(path).points))
            for cnt in range(self._multiple_factor):
                pbar.update()
                _, centroids = fps_sampling_func(points, self._patch_num, self._patch_size, None)
                centroids_group[cnt, :, :, :] = centroids
            self._fixed_fps_centroids_dict[reference_model] = centroids_group


@run_time
def main():
    multiple_factor = 10
    patch_num = 64
    patch_size = 250
    vector_len = 512

    load_log_dir = 'Or150PStanfordForPCTV512_V2'
    raw_dataset_path = r'/workspace/datasets/or150PStanford'
    dis_dataset_path = r'/workspace/datasets/or150PStanford_data5_v2'
    feature_dataset_path = r'/workspace/datasets/or150PStanford_data5_v2_feature_vector'

    dataset = FeatureDatasets(raw_dataset_path, dis_dataset_path, feature_dataset_path, multiple_factor=multiple_factor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''Model loading'''
    # # Use DGCNN
    # arg = argparse.Namespace(emb_dims=1024, k=20, dropout=0.5)
    # feature_extractor = DGCNN(arg, 7).to(device)

    # PCT
    feature_extractor = Point_Transformer(output_channels=vector_len).to(device)

    weight_net = Weight_Net(inp_len=vector_len).to(device)
    feature_extractor.train(), weight_net.train()

    log_dir = '/workspace/Projs/SSL_Multitasking/Logging/log'

    # 测试使用使用 微调后的特征提取器
    # load_log_dir = "SSL_multitask_with_Grad_Norm_50_epoch_or150PStanford"
    checkpoints_name = 'Best_rank_acc.pth'

    checkpoints_path = os.path.join(log_dir, load_log_dir, 'checkpoints/%s' % checkpoints_name)
    checkpoints = torch.load(checkpoints_path)
    feature_extractor.load_state_dict(checkpoints['feature_extractor_state_dict'])
    weight_net.load_state_dict(checkpoints['weight_net_state_dict'])

    # 对fps固定采样点以保证对应patch块选址相同
    dataset.reference_fps_centroids_sampling()
    dataset.save_centroids_dict()

    pbar = tqdm(total=dataloader.__len__())
    pbar.set_description('Generate feature')
    for step, (data, output_path) in enumerate(dataloader):
        pbar.update()

        if isinstance(data, list):
            continue

        feature_vector = np.zeros((multiple_factor * 64, vector_len + 1))
        for i in range(multiple_factor):
            input_data = data[:, i * 64: (i + 1) * 64, :, :]
            patch_fea_vector = feature_extractor(input_data.to(device).view(-1, 3, patch_size))
            weight = weight_net(patch_fea_vector)

            patch_fea_vector = patch_fea_vector.cpu().detach().numpy()
            weight = weight.cpu().detach().numpy()
            patch_fea_with_weight_vector = np.concatenate([patch_fea_vector, weight], axis=1)
            feature_vector[i * 64: (i + 1) * 64, :] = patch_fea_with_weight_vector
            # breakpoint()

        # save
        output_path = output_path[0]
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
            print('Make dir:', os.path.dirname(output_path))

        np.save(output_path, feature_vector)


if __name__ == '__main__':
    main()
    # raw_dataset_path = r'/workspace/datasets/Original_Database_data150'
    # dis_dataset_path = r'/workspace/datasets/data5_adaptive_denoising'
    # feature_dataset_path = r'/workspace/datasets/data5_adaptive_denoising_feature_vector'
    # dataset = FeatureDatasets(raw_dataset_path, dis_dataset_path, feature_dataset_path, multiple_factor=10)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    # data, output_path = dataset[2383]
    #
    # print(data)
