# -*- coding: utf-8 -*-
"""
@author lizheng
@date  12:29
@packageName
@className GMM_modeling
@software PyCharm
@version 1.0.0
@describe TODO
"""
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib as mpl
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold

colors = ["navy", "turquoise", "darkorange"]


def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds


# 检测程序运行时间的函数
def run_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # running_time 取小数点后两位
        running_time = end_time - start_time
        hours, minutes, seconds = format_time(running_time)
        print(
            '\033[33m%s run time: %d hours, %d minutes, %.2f seconds\033[0m'
            % (func.__name__, hours, minutes, seconds))
        return result

    return wrapper


def training_gmm(n_classes, train_index, cov_type, max_iter=20, name=None, vec_len=512,
                 save_path='gmm_estimator_data5_adaptive_denoising_pct.pkl'):
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

    estimator = GaussianMixture(
        n_components=n_classes, covariance_type=cov_type, max_iter=max_iter, random_state=0
    )

    x_train = data[:, :vec_len]
    print('Training GMM data shape:', x_train.shape)

    estimator.fit(x_train)
    root_path = Path('./GMM_Model_params')
    root_path.mkdir(exist_ok=True, parents=True)

    if name is not None:
        file_path = os.path.join(root_path, "%s.pkl" % name)
    else:
        file_path = os.path.join(root_path, save_path)

    print('file_path:', file_path)
    with open(file_path, "wb") as file:
        pickle.dump(estimator, file)
    ''.rstrip()


def loading_trained_gmm(name=None, save_path='gmm_estimator_data5_adaptive_denoising.pkl'):
    if name is not None:
        file_path = "GMM_Model_params/%s.pkl" % name
    else:
        file_path = f"GMM_Model_params/{save_path}"
    with open(file_path, "rb") as file:
        estimator = pickle.load(file)
    return estimator


@run_time
def likelihood_method_main(n_classes, is_train=True, df_info=None):
    multiple_factor = 10
    patch_num = 64
    vec_len = 512
    train_index = pd.read_csv('./Datasets/Boot_file/feature_index_train.csv')
    test_index = pd.read_csv('./Datasets/Boot_file/feature_index_test.csv')
    gmm_model_path = f'gmm_estimator_or150PStanford_data5_v2_pct_k{n_classes}.pkl'
    # gpcd_test_index = pd.read_csv

    # training GMM model
    if is_train:
        training_gmm(n_classes, train_index, 'full', max_iter=100,
                     save_path=gmm_model_path)

    # load estimator
    estimator = loading_trained_gmm(save_path=gmm_model_path)

    total = train_index.__len__() + test_index.__len__()
    pbar = tqdm(total=total)
    # pred on training data
    train_all_likelihood = []
    pbar.set_description('Calculate likelihood on training dateset')
    for step in range(train_index.__len__()):
        pbar.update()
        npy_path = train_index.loc[step, 'feature_vector_path']
        if os.path.exists(npy_path):
            npy_data = np.load(npy_path)
            likelihood_list = []
            for i in range(multiple_factor):
                x = npy_data[i * patch_num: (i + 1) * patch_num, :vec_len]

                # 目前使用平均方法对一组点云的似然函数求平均
                likelihood = np.mean(estimator.score_samples(x))

                likelihood_list.append(likelihood)
            single_pc_likelihood = np.mean(likelihood_list)
            train_all_likelihood.append(single_pc_likelihood)
        else:
            train_all_likelihood.append(-1)
    train_index['likelihood'] = train_all_likelihood

    # pred on testing data
    test_all_likelihood = []
    pbar.set_description('Calculate likelihood on testing dateset')
    for step in range(test_index.__len__()):
        pbar.update()
        npy_path = test_index.loc[step, 'feature_vector_path']
        npy_data = np.load(npy_path)
        likelihood_list = []
        for i in range(multiple_factor):
            x = npy_data[i * patch_num: (i + 1) * patch_num, :vec_len]

            # 目前使用平均方法对一组点云的似然函数求平均
            likelihood = np.mean(estimator.score_samples(x))

            likelihood_list.append(likelihood)
        single_pc_likelihood = np.mean(likelihood_list)
        test_all_likelihood.append(single_pc_likelihood)
    test_index['likelihood'] = test_all_likelihood

    # save result
    print('Saving result in csv file.')
    if df_info:
        train_index = pd.concat([train_index, df_info[0]], axis=1)
        test_index = pd.concat([test_index, df_info[1]], axis=1)

    # 保存
    Path('./result').mkdir(exist_ok=True, parents=True)
    train_index.to_csv(
        f'./result/training likelihood result_n{n_classes}_20241018_pct.csv',
        index=False)
    test_index.to_csv(
        f'./result/testing likelihood result_n{n_classes}_20241018_pct.csv',
        index=False)


def one_dim_quality_calculate(mean: np.ndarray, cov: np.ndarray, mean_d: np.ndarray, cov_d: np.ndarray):
    u = mean_d - mean  # type: np.ndarray
    return np.sqrt(u.T @ np.linalg.inv((cov + cov_d) / 2) @ u)


@run_time
def statistics_distribution_method_main(n_classes, is_train=True, df_info=None):
    multiple_factor = 10
    patch_num = 64
    max_iter = 100
    cov_type = 'full'
    train_index = pd.read_csv('/workspace/Projs/SSL_Multitasking/Datasets/Boot_file/feature_index_train.csv')
    test_index = pd.read_csv('/workspace/Projs/SSL_Multitasking/Datasets/Boot_file/feature_index_test.csv')
    # gpcd_test_index = pd.read_csv

    # training GMM model
    if is_train:
        training_gmm(n_classes, train_index, 'full', max_iter=100)

    # load estimator
    estimator = loading_trained_gmm()

    total = train_index.__len__() + test_index.__len__()
    pbar = tqdm(total=total, colour='#00FFCC')
    index_list = [train_index, test_index]
    for index in index_list:
        quality_all_list = []
        for step in range(index.__len__()):
            pbar.update()
            npy_path = index.loc[step, 'feature_vector_path']
            npy_data = np.load(npy_path)

            # 直接全部建模
            x = npy_data[:, :2048]

            # # 使用分开建模
            # for factor in range(multiple_factor):
            #     x = npy_data[factor * patch_num: (factor + 1) * patch_num, :2048]

            # 对新的一个点云的patch求分布
            single_gmm = GaussianMixture(
                n_components=n_classes, covariance_type=cov_type, max_iter=max_iter, random_state=0
            )
            single_gmm.fit(x)

            # 获取GMM最近高斯关系
            matching_relation_dict = {'new_gmm': [], 'gt_gmm': []}
            matching_calculate_params = []
            gmm_distance_arr = np.zeros((n_classes, n_classes))
            for i in range(n_classes):
                for j in range(n_classes):
                    gmm_distance_arr[i, j] = abs(np.sum(estimator.means_[i] - single_gmm.means_[j]))

            for i in range(n_classes):
                row_index, col_index = np.unravel_index(np.argmin(gmm_distance_arr), gmm_distance_arr.shape)
                matching_relation_dict['gt_gmm'].append(row_index)
                matching_relation_dict['new_gmm'].append(col_index)
                gmm_distance_arr[row_index, :] = np.inf
                gmm_distance_arr[:, col_index] = np.inf
                matching_calculate_params.append(
                    [estimator.means_[row_index], estimator.covariances_[row_index],
                     single_gmm.means_[col_index], single_gmm.covariances_[row_index]]
                )

            quality = np.mean(
                [one_dim_quality_calculate(*matching_calculate_params[i]) for i in range(n_classes)]
            )
            quality_all_list.append(quality)
            # print(f'quality_all_list: {quality_all_list}')
        index['quality'] = quality_all_list

    train_index.to_csv(
        f'./rst_level_with_pred_adaptive_denoising/高斯分布差距法/training likelihood result_n{n_classes}.csv',
        index=False)
    test_index.to_csv(
        f'./rst_level_with_pred_adaptive_denoising/高斯分布差距法/testing likelihood result_n{n_classes}.csv',
        index=False)


if __name__ == '__main__':
    # estimator = loading_trained_gmm()
    # a = np.random.randn(64, 2048)
    # print(estimator.means_.shape)
    # print(estimator.covariances_.shape)
    # train_df_info = pd.read_csv('rst_pmos_with_pred/似然函数法/training likelihood result_n5.csv')
    # train_df_info = train_df_info[['score_mse', 'score_min', 'score_psnr', 'level']]
    # test_df_info = pd.read_csv('rst_pmos_with_pred/似然函数法/testing likelihood result_n5.csv')
    # test_df_info = test_df_info[['score_mse', 'score_min', 'score_psnr', 'level']]

    likelihood_method_main(4, is_train=True, df_info=None)
