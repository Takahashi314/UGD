  # -*- coding: utf-8 -*-
"""
@author lizheng
@date  7:08
@packageName
@className ugd
@software PyCharm
@version 1.0.0
@describe TODO
"""

import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from scipy.special import softmax
from torch.utils.data import Dataset, DataLoader

from gmm import GaussianMixture
from utils import cal_mean_points_edge, run_time
from Model.pct import Point_Transformer
from Model.subtask import Weight_Net
from Extract_ALL_feature_vector import FeatureDatasets
from Datasets.load_dataset import Pairwise_Patch_Point_Cloud_Datasets_Input_PLY
from Datasets.FPS import fps_sampling_func


class UGD:
    def __init__(self, n_components, n_features, infer_repeat_sample,
                 model_log_name, base_log_dir, raw_dataset_path, dis_dataset_path, feature_dataset_path,
                 max_iter=100):
        self.n_components = n_components
        self.n_features = n_features
        self.max_iter = max_iter

        self.infer_repeat_sample = infer_repeat_sample

        self.model_log_name = model_log_name
        self.base_log_dir = base_log_dir
        self.raw_dataset_path = raw_dataset_path
        self.dis_dataset_path = dis_dataset_path
        self.feature_dataset_path = feature_dataset_path

        self.feature_train_csv_path = './boot_file/feature_index_train.csv'
        self.feature_test_csv_path = './boot_file/feature_index_test.csv'

        self.estimator = None
        self.feature_extractor = None
        self.weight_net = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_size, level_len = 0.8, 5
        # train_datasets = Pairwise_Patch_Point_Cloud_Datasets_Input_PLY(
        #     raw_dataset_path, dis_dataset_path, train_size, level_len, pre_load=False
        # )
        # print(train_datasets.train_model_list.__len__())
        # print(train_datasets.test_model_list.__len__())

    def preprocessing_feature_extract(self, patch_num, patch_size):
        multiple_factor = 10
        dataset = FeatureDatasets(raw_dataset_path, dis_dataset_path, feature_dataset_path,
                                  multiple_factor=multiple_factor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        # # Use DGCNN
        # arg = argparse.Namespace(emb_dims=1024, k=20, dropout=0.5)
        # feature_extractor = DGCNN(arg, 7).to(device)

        # PCT
        feature_extractor = Point_Transformer(output_channels=self.n_features).to(self.device)

        weight_net = Weight_Net(inp_len=self.n_features).to(self.device)
        feature_extractor.train(), weight_net.train()
        checkpoints_name = 'Best_rank_acc.pth'
        checkpoints_path = os.path.join(self.base_log_dir, self.model_log_name, 'checkpoints/%s' % checkpoints_name)
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

            feature_vector = np.zeros((multiple_factor * patch_num, self.n_features + 1))
            for i in range(multiple_factor):
                input_data = data[:, i * patch_num: (i + 1) * patch_num, :, :]
                patch_fea_vector = feature_extractor(input_data.to(self.device).view(-1, 3, patch_size))
                weight = weight_net(patch_fea_vector)

                patch_fea_vector = patch_fea_vector.cpu().detach().numpy()
                weight = weight.cpu().detach().numpy()
                patch_fea_with_weight_vector = np.concatenate([patch_fea_vector, weight], axis=1)
                # patch_fea_with_weight_vector = patch_fea_vector
                feature_vector[i * patch_num: (i + 1) * patch_num, :] = patch_fea_with_weight_vector
                # breakpoint()

            # save
            output_path = output_path[0]
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
                print('Make dir:', os.path.dirname(output_path))

            np.save(output_path, feature_vector)

    @run_time
    def training_gmm(self, save_params_path, use_cuda=False, sample_radio=0.8, eps=1e-5):
        # sample_radio = 0.8
        # eps = 1e-5  # 1e-6
        train_index = pd.read_csv(self.feature_train_csv_path)
        gt_index = train_index[train_index['type'] == 'GT']
        gt_index.reset_index(inplace=True)
        data = None

        pbar = tqdm(total=gt_index.__len__())
        pbar.set_description('Load GMM training  data')
        for i in range(gt_index.__len__()):
            pbar.update()
            npy_path = gt_index.loc[i, 'feature_vector_path']
            if data is None:
                data = np.load(npy_path)
            else:
                data = np.concatenate([data, np.load(npy_path)], axis=0)

        max_date_inputs = int(sample_radio * data.shape[0])
        x_train = torch.tensor(data[:max_date_inputs, :self.n_features])
        if use_cuda:
            x_train = x_train.to(self.device)
        estimator = GaussianMixture(n_components=self.n_components, n_features=self.n_features, eps=eps)
        with torch.no_grad():
            estimator.fit(x_train, n_iter=self.max_iter)

        gmm_state_dict = {
            'pi': estimator.pi.cpu(),
            'mu': estimator.mu.cpu(),
            'var': estimator.var.cpu(),
        }

        Path(save_params_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(gmm_state_dict, save_params_path)
        print(f"Save training set : {save_params_path}")
        return save_params_path

    def load_gmm_params(self, params_path, use_cuda=False):
        gmm_state_dict = torch.load(params_path, map_location='cpu')
        estimator = GaussianMixture(n_components=self.n_components, n_features=self.n_features)
        if not use_cuda:
            estimator.pi.data = gmm_state_dict['pi']
            estimator.mu.data = gmm_state_dict['mu']
            estimator.var.data = gmm_state_dict['var']
        else:
            estimator.pi.data = gmm_state_dict['pi'].to(self.device)
            estimator.mu.data = gmm_state_dict['mu'].to(self.device)
            estimator.var.data = gmm_state_dict['var'].to(self.device)
        self.estimator = estimator
        return estimator

    def feature_encoding(self, pc_paths, scale=None, patch_num=64, patch_size=250):
        pc_list = [np.asarray(o3d.io.read_point_cloud(path).points) for path in pc_paths]
        pc_list = [torch.from_numpy(item).float().unsqueeze(0) for item in pc_list]

        if scale:
            pc_list = [item / scale for item in pc_list]

        # 获取锚点
        reference_pc = pc_list[0]
        _, centroids = fps_sampling_func(reference_pc, patch_num, patch_size, None)

        pc_patch_list = []
        for step, item in enumerate(pc_list):
            # print(pc_path_list[step])
            points_with_patch, _ = fps_sampling_func(item, patch_num, patch_size, centroids)
            pc_patch_list.append(points_with_patch)

        patch_encoding_vector_list = []
        for item in pc_patch_list:
            patch_fea_vector = self.feature_extractor(item.to(self.device).view(-1, 3, patch_size))
            weight = self.weight_net(patch_fea_vector)
            patch_fea_vector = patch_fea_vector.cpu().detach().numpy()
            weight = weight.cpu().detach().numpy()
            patch_encoding_vector = np.concatenate([patch_fea_vector, weight], axis=1)
            patch_encoding_vector_list.append(patch_encoding_vector)
        return patch_encoding_vector_list

    def load_feature_extractor(self):
        """Model loading"""
        # # Use DGCNN
        # arg = argparse.Namespace(emb_dims=1024, k=20, dropout=0.5)
        # feature_extractor = DGCNN(arg, 7).to(device)

        # PCT
        feature_extractor = Point_Transformer(output_channels=self.n_features).to(self.device)

        weight_net = Weight_Net(inp_len=self.n_features).to(self.device)
        feature_extractor.train(), weight_net.train()

        checkpoints_name = 'Best_rank_acc.pth'
        checkpoints_path = os.path.join(self.base_log_dir, self.model_log_name, 'checkpoints/%s' % checkpoints_name)
        checkpoints = torch.load(checkpoints_path)
        feature_extractor.load_state_dict(checkpoints['feature_extractor_state_dict'])
        weight_net.load_state_dict(checkpoints['weight_net_state_dict'])
        feature_extractor.train(), weight_net.train()
        self.feature_extractor = feature_extractor
        self.weight_net = weight_net

    def calculate_ugd(self, pc_paths: list, use_cuda=False, patch_num=64, patch_size=250):
        log_space = 2
        scale = cal_mean_points_edge(pc_paths[0]) / 0.0076
        score_dict = {path: [] for path in pc_paths}
        for i in range(self.infer_repeat_sample):
            if i % log_space == 0:
                print(f"Calculate UGD Repeat Sample of {i} / {self.infer_repeat_sample}")

            patch_encoding_vector_list = self.feature_encoding(pc_paths, scale, patch_num=patch_num,
                                                               patch_size=patch_size)
            for step, patch_encoding_vector in enumerate(patch_encoding_vector_list):
                data = patch_encoding_vector[:, :self.n_features]
                w = softmax(patch_encoding_vector[:, self.n_features])
                data = torch.tensor(data).double()
                if use_cuda:
                    data = data.to(self.device)
                score_dict[pc_paths[step]].append(self.estimator.score_samples(data).mean().cpu().item())
        ugds = [np.mean(item) for item in score_dict.values()]
        return ugds

    def calculate_ugd_directly_feature(self, features_path, use_cuda=False, patch_num=64):
        features = [np.load(item) for item in features_path]
        train_repeat = features[0].shape[0] // patch_num
        log_space = 2
        score_dict = {path: [] for path in features_path}
        for i in range(train_repeat):
            if i % log_space == 0:
                print(f"Calculate UGD Repeat Sample of {i} / {self.infer_repeat_sample}")

            patch_encoding_vector_list = [item[i * patch_num: (i + 1) * patch_num, :] for item in features]
            for step, patch_encoding_vector in enumerate(patch_encoding_vector_list):
                data = patch_encoding_vector[:, :self.n_features]
                w = softmax(patch_encoding_vector[:, self.n_features])
                data = torch.tensor(data).double()
                if use_cuda:
                    data = data.to(self.device)
                score_dict[features_path[step]].append(self.estimator.score_samples(data).mean().cpu().item())
        ugds = [np.mean(item) for item in score_dict.values()]
        return ugds

    @staticmethod
    def ugd_relative_gain(reference_ugd, target_ugd):
        diff = reference_ugd - target_ugd
        return np.arctan(diff)

    def pairwise_accuracy(self, ugds, true_labels):
        n = len(ugds)
        correct_num = 0
        pairwise_num = 0
        sort_ugds = [item[0] for item in sorted(list(zip(ugds, true_labels)), key=lambda x: x[1])]
        for i in range(n - 1):
            for j in range(i + 1, n):
                pairwise_num += 1
                relative_gain = self.ugd_relative_gain(sort_ugds[i], sort_ugds[j])
                if relative_gain >= 0:
                    correct_num += 1
        return correct_num / pairwise_num, correct_num, pairwise_num

    @run_time
    def validate_test_set(self, result_df_path, use_cuda=True, patch_num=64, patch_size=250):
        test_index_df = pd.read_csv(self.feature_test_csv_path)
        # models_name = list(test_index_df['reference_model'].value_counts().index)
        models_name = [item for item in list(test_index_df['reference_model'].value_counts().index)
                       if item.startswith('raw_')]
        types_name = [item for item in test_index_df['type'].value_counts().index if item != 'GT']

        data = {key: [] for key in types_name}
        pbar = tqdm(total=len(models_name) * len(types_name), desc="Validate on test dataset")
        for i, model_name in enumerate(models_name):
            model_df = test_index_df[test_index_df['reference_model'] == model_name]
            for dis_type in types_name:
                list_df = model_df[(model_df['type'] == dis_type) | (model_df['type'] == 'GT')]
                pc_paths = list(list_df['ply_file_path'])
                ugds = self.calculate_ugd(pc_paths, use_cuda=use_cuda, patch_num=patch_num, patch_size=patch_size)
                true_labels = list(list_df['level'])  # 0 为最佳质量 标签越大质量越差
                acc, correct_num, pairwise_num = self.pairwise_accuracy(ugds, true_labels)
                data[dis_type].append(acc)
                pbar.update(1)

        result_df_path = Path(result_df_path)
        result_df_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(data, columns=types_name, index=models_name)
        df.to_csv(str(result_df_path))
        print(f"Save result to {result_df_path}")
        return df

    @run_time
    def validate_train_set(self, result_df_path, use_cuda=True, patch_num=64):
        train_index_df = pd.read_csv(self.feature_train_csv_path)
        # models_name = list(train_index_df['reference_model'].value_counts().index)
        models_name = [item for item in list(train_index_df['reference_model'].value_counts().index)
                       if item.startswith('raw_')]
        types_name = [item for item in train_index_df['type'].value_counts().index if item != 'GT']

        data = {key: [] for key in types_name}
        pbar = tqdm(total=len(models_name) * len(types_name), desc="Validate on test dataset")
        for i, model_name in enumerate(models_name):
            model_df = train_index_df[train_index_df['reference_model'] == model_name]
            for dis_type in types_name:
                list_df = model_df[(model_df['type'] == dis_type) | (model_df['type'] == 'GT')]
                feature_paths = list(list_df['feature_vector_path'])
                ugds = self.calculate_ugd_directly_feature(feature_paths, use_cuda=use_cuda, patch_num=patch_num)
                true_labels = list(list_df['level'])  # 0 为最佳质量 标签越大质量越差
                acc, correct_num, pairwise_num = self.pairwise_accuracy(ugds, true_labels)
                data[dis_type].append(acc)
                pbar.update(1)

        result_df_path = Path(result_df_path)
        result_df_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(data, columns=types_name, index=models_name)
        df.to_csv(str(result_df_path))


def auto_run_components(test_n=[1, 2, 3, 5, 8, 12, 16, 18, 24, 32, 40, 50]):
    for n in test_n:
        model_log_name = 'Or150PStanfordForPCTV512_V2'  # 特征编码器 LOG 文件名
        base_log_dir = './Logging/log'  # 模型保存位置
        raw_dataset_path = r'../datasets/or150PStanford'  # 原始点云存放位置
        dis_dataset_path = r'../datasets/or150PStanford_data5_v2'  # 失真点云存放位置
        feature_dataset_path = r'../datasets/or150PStanford_data5_v2_feature_vector'  # 预生成特征存放位置

        # n_components_list = [32, 40]
        n_components = n
        patch_num = 64
        patch_size = 250
        n_features = 512
        infer_repeat_sample = 20
        # for n_components in n_components_list:
        #     print(f'开始处理：n_component={n_components}')

        ugd = UGD(n_components, n_features, infer_repeat_sample,
                  model_log_name, base_log_dir, raw_dataset_path, dis_dataset_path, feature_dataset_path)

        # 进行特征预处理 预提取特征以建模使用
        ugd.preprocessing_feature_extract(patch_num, patch_size)

        # # GMM 建模
        saved_params_path = f'./gmm_params/gmm_estimator_{n_components}_{patch_num}_{patch_size}.pth'
        if not os.path.exists(saved_params_path):
            sample_radio = 1
            if n > 18:
                sample_radio = 0.6
            saved_params_path = ugd.training_gmm(use_cuda=True, sample_radio=sample_radio, eps=1e-5)
        # print(saved_params_path)

        # UGD 计算
        # n_components_patch_num_patch_size

        ugd.load_gmm_params(saved_params_path, use_cuda=True)
        ugd.load_feature_extractor()
        # # 测试集上进行测试结果保存
        result_df_path = f"./result/experiment03_test_{n_components}_{patch_num}_{patch_size}_result_v2.csv"
        ugd.validate_test_set(result_df_path, use_cuda=True, patch_num=patch_num)
        # # 训练集上进行测试结果保存
        result_df_path = f"./result/experiment03_train_{n_components}_{patch_num}_{patch_size}_result_v2.csv"

        ugd.validate_train_set(result_df_path, use_cuda=True, patch_num=patch_num)


def fps_stability_test(ugd):
    # UGD 计算
    # n_components_patch_num_patch_size
    load_params_path = f'./gmm_params/gmm_estimator_{n_components}_{64}_{250}.pth'
    ugd.load_gmm_params(load_params_path, use_cuda=True)
    ugd.load_feature_extractor()

    folder = 'AE_FPS'
    files = [
        'bunny.ply',
        'bunny_D02_L01.ply',
        'bunny_D02_L02.ply',
        'bunny_D02_L03.ply',
        'bunny_D02_L04.ply'
    ]

    sample_times = 20
    use_cuda = True

    pc_paths = [os.path.join(folder, item) for item in files]
    for i in range(sample_times):
        ugds = ugd.calculate_ugd(pc_paths, use_cuda=use_cuda, patch_num=patch_num, patch_size=patch_size)
        print(ugds)
        breakpoint()


if __name__ == '__main__':
    # model_log_name = 'Or150PStanfordForPCTV512_V2'  # 特征编码器 LOG 文件名
    # base_log_dir = '/workspace/Projs/SSL_Multitasking/Logging/log'  # 模型保存位置
    # raw_dataset_path = r'/workspace/datasets/or150PStanford'  # 原始点云存放位置
    # dis_dataset_path = r'/workspace/datasets/or150PStanford_data5_v2'  # 失真点云存放位置
    # feature_dataset_path = r'/workspace/datasets/or150PStanford_data5_v2_feature_vector'  # 预生成特征存放位置


    base_log_dir = '/workspace/Projs/SSL_Multitasking/Logging/log'
    model_log_name = 'Or150PStanfordForPCTV512_V2'
    raw_dataset_path = r'/workspace/datasets/or150PStanford'  # 原始点云存放位置
    dis_dataset_path = r'/workspace/datasets/or150PStanford_data5_v2'  # 失真点云存放位置
    feature_dataset_path = r'/workspace/datasets/or150PStanford_data5_feature_vector_T'

    # n_components = 5
    patch_num = 64
    patch_size = 250
    n_features = 512
    infer_repeat_sample = 10

    total_df = None
    iteration_list = list(range(1, 3))
    for i, n_components in enumerate(iteration_list):
        ugd = UGD(n_components, n_features, infer_repeat_sample,
                  model_log_name, base_log_dir, raw_dataset_path, dis_dataset_path, feature_dataset_path)

        # # 进行特征预处理 预提取特征以建模使用
        # ugd.preprocessing_feature_extract(patch_num, patch_size)

        # GMM 建模
        save_params_path = f'./gmm_params/gmm_estimator_{n_components}_{64}_{250}.pth'
        # sample_radio = -0.025 * n_components + 1
        sample_radio = 0.05
        saved_params_path = ugd.training_gmm(save_params_path, use_cuda=True, sample_radio=sample_radio)
        print(saved_params_path)

        # UGD 计算
        # n_components_patch_num_patch_size
        load_params_path = f'./gmm_params/gmm_estimator_{n_components}_{64}_{250}.pth'

        ugd.load_gmm_params(load_params_path, use_cuda=True)
        ugd.load_feature_extractor()
        # # 测试集上进行测试结果保存
        result_df_path = f"./test_result_k{n_components}.csv"
        rst_df = ugd.validate_test_set(result_df_path, use_cuda=True, patch_num=patch_num, patch_size=patch_size)

        series = rst_df.mean()
        if total_df is None:
            total_df = series.to_frame().T
        else:
            total_df = total_df.append(series.to_frame().T)

        Path('./components_analysis').mkdir(parents=True, exist_ok=True)
        total_df.to_csv('./components_analysis/n_components_analysis.csv')
    total_df.index = iteration_list
    total_df.to_csv('./components_analysis/n_components_analysis.csv')
    # # # # 训练集上进行测试结果保存
    # # result_df_path = f"./result/experiment03_train_{n_components}_{patch_num}_{patch_size}_result_v2.csv"
    # # ugd.validate_train_set(result_df_path, use_cuda=True)
    # print(f'patch_size={patch_size}处理完毕')

    # ----------------------------------------------------------
    # fps_stability_test(ugd)
