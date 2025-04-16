# -*- coding: utf-8 -*-
"""
@author lizheng
@date 2023年01月27日 20:39:18
@packageName 
@className load_mos_dataset
@version 1.0.0
@describe
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class G_PCD_MOS_Dataset(Dataset):
    def __init__(self, dataset_path1, dataset_path2, subj_mos_path):
        self.dataset_path1 = dataset_path1
        self.dataset_path2 = dataset_path2
        self.subj_mos_path = subj_mos_path
        self.subj_df = pd.read_csv(self.subj_mos_path)

    def __len__(self):
        return self.subj_df.__len__()

    def __getitem__(self, item):
        if item < 25:
            dataset_path = self.dataset_path1
        else:
            dataset_path = self.dataset_path2
        file_path = os.path.join(dataset_path, '%s.pth' % self.subj_df.iloc[item]['stimulus'])
        # print(file_path)
        data = torch.load(file_path)
        data = data.permute(2, 0, 1).view(3, -1)
        return data, self.subj_df.iloc[item]['MOS']


class Data10_MOS_Dataset(Dataset):
    def __init__(self, dataset_path, subj_mos_path):
        self.dataset_path = dataset_path
        self.subj_mos_path = subj_mos_path
        self.subj_df = pd.read_csv(self.subj_mos_path)

        # # save new csv
        # self.subj_df = pd.DataFrame(columns=['path', 'mos'])
        # self.load_subj_df()
        # self.subj_df.to_csv(self.subj_mos_path.replace('.txt', '.csv'), index=False)

    def __len__(self):
        return self.subj_df.__len__()

    def __getitem__(self, item):
        file_path = self.subj_df.iloc[item]['path']
        # print(file_path)
        data = torch.load(file_path)
        data = data.permute(2, 0, 1).view(3, -1)
        return data, self.subj_df.iloc[item]['mos']

    def load_subj_df(self):
        with open(self.subj_mos_path) as f:
            f_str = f.read()
        line_list = f_str.split('\n')
        line_list.pop()
        for line in line_list:
            line_split = line.split(' ')
            self.subj_df.loc[len(self.subj_df)] = [line_split[0], float(line_split[1])]


def random_split(load_file_path, training_rate):
    original_df = pd.read_csv(load_file_path)
    random_index = np.arange(original_df.__len__())
    np.random.shuffle(random_index)
    training_df = pd.DataFrame(columns=original_df.columns)
    for i in range(int(training_rate * original_df.__len__())):
        training_df.loc[i] = original_df.loc[random_index[i]]
    training_df.to_csv('%s_training_rate_%d.csv' % (load_file_path[:-4], int(training_rate * 100)), index=False)


if __name__ == '__main__':
    subj_path = r'./Boot_file/listwise_train_03_txt_add_score.csv'
    #
    # data_path = r'E:\homegate\R_Quality_Assessment\Dataset\Data10_fixed_fps_64_128'
    # Datasets = Data10_MOS_Dataset(data_path, subj_path)
    # dataloader = DataLoader(Datasets, batch_size=6, shuffle=False, pin_memory=True)
    #
    # for step, (x, y) in enumerate(dataloader):
    #     print(step)
    #
    # print(x.shape, y)

    # subj_path = r'./Boot_file/subj_desktop_dsis_joint.csv'
    # data_path1 = r'E:\homegate\R_Quality_Assessment\Dataset\G-PCD\stimuli\D01_fixed_fps'
    # data_path2 = r'E:\homegate\R_Quality_Assessment\Dataset\G-PCD\stimuli\D02_fixed_fps'
    # Datasets = G_PCD_MOS_Dataset(data_path1, data_path2, subj_path)

    random_split(subj_path, 0.5)
