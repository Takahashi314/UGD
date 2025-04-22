# -*- coding: utf-8 -*-
"""
@author lizheng
@date  0:18
@packageName
@className Multitask with grad_norm
@software PyCharm
@version 1.0.0
@describe
"""
import pandas as pd
import torch
# import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from Datasets.load_dataset import Pairwise_Patch_Point_Cloud_Datasets_Input_PLY
from torch.utils.data import DataLoader
from Logging.train_logging import Train_loging
from Model.dgcnn_model import DGCNN
from Model.pct import Point_Transformer
import argparse
from Uploading.Email.email_sending import sending_emil
# from Model.network_liyong_dgcnn import double_fusion
import time
import datetime
from torch.cuda.amp import autocast, GradScaler
import os
from Model.subtask import Classification_Type, Weight_Net, Pair_Net
from Model.ranknet import refresh_pairwise_correct_list

from Model.GradNorm import GradNorm


def train(name, idx):
    num_workers = 12
    alpha = 0.5
    training_epoches = 10 * 2
    epoch_multiplier = 105
    step_divisor = 630  # 630 为10%
    batch_size = 4
    patch_number = 64
    patch_size = 250
    theta = 0.5

    # log_dir_name = 'SSL_multitask_with_Grad_Norm_50_epoch_adaptive_denoising'
    log_dir_name = f'Or150PStanfordForPCTV512_V3_AE_{name}'

    # raw_dataset_path = r'/workspace/datasets/Original_Database_data150'
    # dis_dataset_path = r'/workspace/datasets/data5_adaptive_denoising'

    raw_dataset_path = r'D:\lz\dataset\or150PStanford'
    dis_dataset_path = r'D:\lz\dataset\or150PStanford_data5_v2'

    '''Data loading'''
    print()
    train_datasets = Pairwise_Patch_Point_Cloud_Datasets_Input_PLY(raw_dataset_path, dis_dataset_path, train_size=0.8,
                                                                   level_len=5, train=True, patch_size=patch_size,
                                                                   multiplier=epoch_multiplier, random_split=False)
    test_datasets = Pairwise_Patch_Point_Cloud_Datasets_Input_PLY(raw_dataset_path, dis_dataset_path, train_size=0.8,
                                                                  level_len=5, train=False, patch_size=patch_size,
                                                                  multiplier=epoch_multiplier, random_split=False)
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''Model loading'''
    # # Use DGCNN
    # arg = argparse.Namespace(emb_dims=1024, k=20, dropout=0.5)
    # feature_extractor = DGCNN(arg, 7).to(device)

    # PCT
    vector_len = 512
    feature_extractor = Point_Transformer(output_channels=vector_len).to(device)

    classifier = Classification_Type(6, inp_len=vector_len).to(device)
    weight_net = Weight_Net(inp_len=vector_len).to(device)
    pairnet = Pair_Net(batch_size, patch_number, inp_len=vector_len).to(device)
    gradnorm = GradNorm(classifier, pairnet, device).to(device)

    '''parameter setting'''
    optimizer = torch.optim.AdamW(
        [{'params': feature_extractor.parameters()},
         {'params': weight_net.parameters()},
         {'params': gradnorm.parameters()}],
        lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    scaler = GradScaler()

    logging = Train_loging(log_dir_name, training_epoches)
    load_epoch = 0
    logging.csv_init(['loss', 'classifier_train_loss', 'pairwise_train_loss', 'kl_loss',
                      'classifier_train_acc', 'pairwise_train_acc',
                      'task1_weights', 'task2_weights', 'task3_weights'], load_epoch)
    logging.load_epoch = load_epoch

    logging.start_training('Start training progress...')

    pairwise_best_instance_acc = 0.0
    best_loss = 200.0

    # # loading training
    # checkpoints = torch.load(str(logging.checkpoints_dir) + '/log_epoch_%d.pth' % 6)
    # pairnet.load_state_dict(checkpoints['pairnet_state_dict'])
    # load_epoch = checkpoints['epoch']

    # pbar
    pbar_training = tqdm(total=(training_epoches - load_epoch) * train_loader.__len__())
    pbar_testing = tqdm(total=(training_epoches - load_epoch) * test_loader.__len__())

    for epoch in range(load_epoch, training_epoches):
        logging.start_single_epoch(epoch)
        torch.cuda.empty_cache()
        # loss = 0
        last_loss = 0
        last_task_loss = 0
        all_loss_list = []
        batch_loss_list = []
        batch_loss_classification_list = []
        batch_loss_distribution_list = []
        batch_loss_rank_list = []
        batch_loss_kl_list = []
        all_mean_correct = []
        batch_mean_correct = []
        batch_mse_distribution_list = []
        pairwise_correct_list = np.empty(0, dtype=bool)
        batch_pairwise_correct_list = np.empty(0, dtype=bool)

        feature_extractor.train(), classifier.train(), weight_net.train(), pairnet.train()
        for step, (data_pth_list, distortion_type_label,
                   distortion_level_label_list, pairwise_label, alpha_distribution) in enumerate(train_loader):
            pbar_training.set_description('Epoch %d Training:' % epoch)
            pbar_training.update()
            pairwise_label = pairwise_label.to(device).to(torch.float)
            alpha_distribution = alpha_distribution.to(device)

            data_pth_list[0] = data_pth_list[0].permute(0, 1, 3, 2)
            data_pth_list[1] = data_pth_list[1].permute(0, 1, 3, 2)

            optimizer.zero_grad()
            with autocast():
                # 特征值和权重
                patch_fea_vector1 = feature_extractor(data_pth_list[0].to(device).view(-1, 3, patch_size))
                patch_fea_vector2 = feature_extractor(data_pth_list[1].to(device).view(-1, 3, patch_size))
                weight_1 = weight_net(patch_fea_vector1)
                weight_2 = weight_net(patch_fea_vector2)

                # print(f"patch_fea_vector1:\n{patch_fea_vector1}")
                if torch.isnan(torch.sum(patch_fea_vector1)) or torch.isnan(torch.sum(patch_fea_vector2)):
                    print(f"patch_fea_vector, patch_fea_vector2 is nan")
                    breakpoint()
                model_fea_vector1 = torch.mul(patch_fea_vector1, weight_1). \
                    view(batch_size, patch_number, vector_len).mean(dim=1)
                model_fea_vector2 = torch.mul(patch_fea_vector2, weight_2). \
                    view(batch_size, patch_number, vector_len).mean(dim=1)

                task_loss, y_pred_type1, y_pred_type2, dist1, dist2, distri_pred1, distri_pred2 = \
                    gradnorm(patch_fea_vector1, patch_fea_vector2, weight_1, weight_2,
                             model_fea_vector1, model_fea_vector2,
                             distortion_type_label, pairwise_label, alpha_distribution)

                # task loss ----> loss_classification, loss_rank, distri_loss
                if(idx<=2):
                    loss = task_loss[idx]
                else:
                    loss = task_loss[0]+task_loss[1]+task_loss[2]

                # # GradNorm
                # task_loss *= scaler.get_scale()
                # weighted_task_loss = torch.mul(gradnorm.weights, task_loss)
                # if step == 0:
                #     # set L(0)
                #     if torch.cuda.is_available():
                #         initial_task_loss = task_loss.data.cpu()
                #     else:
                #         initial_task_loss = task_loss.data
                #     initial_task_loss = initial_task_loss.numpy()
                #
                # # get the total loss
                # loss = torch.sum(weighted_task_loss / scaler.get_scale())
                #
                # clear the gradients
                optimizer.zero_grad()
                scaler.scale(loss).backward(retain_graph=True)
                #
                # # print('loss: ', loss)
                # # print('scaler.scale(loss): ', scaler.scale(loss), type(scaler.scale(loss)))
                # # print('scaler._scale', scaler._scale)
                # # print('scaler.get_scale(): ', scaler.get_scale())
                # # print('scaler._scale * loss: ', scaler._scale * loss)
                # # exit()
                # # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
                # gradnorm.weights.grad.data = gradnorm.weights.grad.data * 0.0
                # # print('gradnorm.weights', gradnorm.weights)
                #
                # # get layer of shared weights
                # w = feature_extractor.get_last_shared_layer()
                # # get the gradient norms for each of the tasks
                # norms = []
                # # 归一化权重
                # n_weights = gradnorm.weights / torch.sum(gradnorm.weights)
                # for i in range(3):
                #     # get the gradient of this task loss with respect to the shared parameters
                #     gygw = torch.autograd.grad(task_loss[i], w.parameters(), retain_graph=True)
                #     # compute the norm
                #     norms.append(torch.norm(torch.mul(n_weights[i], gygw[0])))
                # norms = torch.stack(norms)
                #
                # # compute the inverse training rate r_i(t)
                # loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
                # # r_i(t)
                # inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                # # compute the mean norm \tilde{G}_w(t)
                # mean_norm = np.mean(norms.data.cpu().numpy())
                # # compute the GradNorm loss
                # constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
                # constant_term = constant_term.cuda()
                # # this is the GradNorm loss itself
                # grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                # # print('mean_norm: ', mean_norm, ' grad_norm_loss: ', grad_norm_loss, )
                # # print('inverse_train_rate: ', inverse_train_rate, 'n_weights: ', n_weights)
                # # compute the gradient for the weights
                # gradnorm.weights.grad = torch.autograd.grad(grad_norm_loss, gradnorm.weights)[0]
                # task_loss /= scaler.get_scale()

            all_loss_list.append(loss.detach().cpu().item())
            batch_loss_list.append(loss.detach().cpu().item())

            batch_loss_classification_list.append(task_loss[0].detach().cpu().item() * theta)
            batch_loss_rank_list.append(task_loss[1].detach().cpu().item())
            batch_loss_kl_list.append(task_loss[2].detach().cpu().item())
            batch_loss_distribution_list.append(task_loss[2].detach().cpu().item() * theta)

            # 分类精度 hard classification
            pred_choice1 = y_pred_type1.data.max(1)[1].cpu()
            correct1 = pred_choice1.eq(distortion_type_label.long().data).cpu().sum()
            all_mean_correct.append(correct1.item() / float(distortion_type_label.detach().size()[0]))
            batch_mean_correct.append(correct1.item() / float(distortion_type_label.detach().size()[0]))

            pred_choice2 = y_pred_type2.data.max(1)[1].cpu()
            correct2 = pred_choice2.eq(distortion_type_label.long().data).cpu().sum()
            all_mean_correct.append(correct2.item() / float(distortion_type_label.detach().size()[0]))
            batch_mean_correct.append(correct2.item() / float(distortion_type_label.detach().size()[0]))

            # MSE of distortion
            mse1 = torch.nn.functional.mse_loss(alpha_distribution, distri_pred1)
            mse2 = torch.nn.functional.mse_loss(alpha_distribution, distri_pred2)
            mse = (mse1 + mse2) / 2
            batch_mse_distribution_list.append(mse.detach().cpu().item())

            # 排序正确率
            pairwise_correct_list = refresh_pairwise_correct_list(pairwise_correct_list, dist1, dist2, pairwise_label)
            batch_pairwise_correct_list = refresh_pairwise_correct_list(batch_pairwise_correct_list, dist1, dist2,
                                                                        pairwise_label)

            scaler.step(optimizer)
            scaler.update()

            # if torch.isnan(loss) or loss < 0.01:
            #     print('Find training loss nan')
            #     print('last loss: ', last_loss)
            #     print('gradnorm.weights: ', gradnorm.weights + 0.00)
            #     print('n_weights: ', n_weights)
            #     print('inverse_train_rate: ', inverse_train_rate)
            #     print('loss_ratio: ', loss_ratio)
            #     print('task_loss: ', task_loss)
            #     print('last_task_loss: ', last_task_loss)
            #     print('initial_task_loss: ', initial_task_loss)
            #     print('grad_norm_loss: ', grad_norm_loss)
            #     print('gygw: ', gygw)
            #     print('norms:', norms)
            #     print('mean_norm: ', mean_norm)
            #     print('constant_term: ', constant_term)
            #     print('scaler.get_scale()', scaler.get_scale())
            #     print('gradnorm.weights.grad: ', gradnorm.weights.grad)
            #     print('\n')
            #     breakpoint()
            #     exit(103)

            del y_pred_type1, y_pred_type2, dist1, dist2
            torch.cuda.empty_cache()

            last_task_loss = task_loss.detach().cpu()
            last_loss = loss.detach().cpu().item()

            if (step + 1) % step_divisor == 0:
                batch_mean_correct_acc = np.mean(batch_mean_correct)
                batch_distri_mse = np.mean(batch_mse_distribution_list)
                batch_pairwise_train_instance_acc = \
                    batch_pairwise_correct_list.sum() / batch_pairwise_correct_list.__len__()

                # print('gradnorm.weights: ', gradnorm.weights)
                # print('n_weights: ', n_weights)
                # print('inverse_train_rate: ', inverse_train_rate)
                # print('loss_ratio: ', loss_ratio)
                # print('task_loss: ', task_loss)
                # print('initial_task_loss: ', initial_task_loss)
                # print('grad_norm_loss: ', grad_norm_loss)
                # # print('gygw: ', gygw)
                # # print('norms:', norms)
                # # print('mean_norm: ', mean_norm)
                # # print('constant_term: ', constant_term)
                # print('scaler.get_scale()', scaler.get_scale())
                # # print('gradnorm.weights.grad: ', gradnorm.weights.grad)
                # print('\n')

                logging.log_string('Train loss: %f' % np.mean(batch_loss_list))
                logging.log_string('Train loss of Classification: %f' % np.mean(batch_loss_classification_list))
                logging.log_string('Train loss of Rank: %f' % np.mean(batch_loss_rank_list))
                logging.log_string('Train loss of KL: %f' % np.mean(batch_loss_kl_list))
                logging.log_string('Train loss of distortion: %f' % np.mean(batch_mse_distribution_list))

                logging.log_string('Classification: Train Instance Accuracy: %f' % batch_mean_correct_acc)
                logging.log_string('Distribution: Train Instance MSE: %f' % batch_distri_mse)
                logging.log_string('Pairwise: Train Instance Accuracy: %f' % batch_pairwise_train_instance_acc)

                # logging.log_string('Task weights:', n_weights)

                index = epoch * train_loader.__len__() // step_divisor + step // step_divisor
                logging.save_data_csv([np.mean(batch_loss_list),
                                       np.mean(batch_loss_classification_list), np.mean(batch_loss_rank_list),
                                       np.mean(batch_mse_distribution_list),
                                       batch_mean_correct_acc, batch_pairwise_train_instance_acc,
                                       0, 0, 0], index)
                state = {
                    'index': index,
                    'feature_extractor_state_dict': feature_extractor.state_dict(),
                    'grad_norm_state_dict': gradnorm.state_dict(),
                    'weight_net_state_dict': weight_net.state_dict(),
                    'loss': np.mean(all_loss_list),
                }
                logging.save_check_points(state['index'], state)

                batch_loss_list = []
                batch_mean_correct = []
                batch_loss_list = []
                batch_mse_distribution_list = []
                batch_loss_classification_list = []
                batch_loss_distribution_list = []
                batch_loss_rank_list = []
                batch_loss_kl_list = []
                batch_pairwise_correct_list = np.empty(0, dtype=bool)

        pairwise_train_instance_acc = pairwise_correct_list.sum() / pairwise_correct_list.__len__()
        all_mean_correct_acc = np.mean(all_mean_correct)
        all_mean_loss = np.mean(all_loss_list)
        step_lr.step()

        logging.log_string('Classification: Train Instance Accuracy: %f' % all_mean_correct_acc)
        logging.log_string('Pairwise: Train Instance Accuracy: %f' % pairwise_train_instance_acc)
        logging.log_string('Train loss: %f' % all_mean_loss)

        # test
        with torch.no_grad():
            all_mean_correct = []
            classification_best_instance_acc = 0
            pairwise_correct_list = np.empty(0, dtype=bool)
            for step, (data_pth_list, distortion_type_label,
                       distortion_level_label_list, pairwise_label, alpha_distribution) in enumerate(test_loader):
                pbar_testing.set_description('Epoch %d Testing:' % epoch)
                pbar_testing.update()
                pairwise_label = pairwise_label.to(device).to(torch.float)

                data_pth_list[0] = data_pth_list[0].permute(0, 1, 3, 2)
                data_pth_list[1] = data_pth_list[1].permute(0, 1, 3, 2)

                with autocast():
                    patch_fea_vector1 = feature_extractor(data_pth_list[0].to(device).view(-1, 3, patch_size))
                    patch_fea_vector2 = feature_extractor(data_pth_list[1].to(device).view(-1, 3, patch_size))
                    weight_1 = weight_net(patch_fea_vector1)
                    weight_2 = weight_net(patch_fea_vector2)
                    model_fea_vector1 = torch.mul(patch_fea_vector1, weight_1). \
                        view(batch_size, patch_number, vector_len).mean(dim=1)
                    model_fea_vector2 = torch.mul(patch_fea_vector2, weight_2). \
                        view(batch_size, patch_number, vector_len).mean(dim=1)

                    # 排序任务
                    y_pred_type1, distri_pred1 = classifier(model_fea_vector1)
                    y_pred_type2, distri_pred2 = classifier(model_fea_vector2)

                    # 排序任务
                    dist1, dist2, out, diff = pairnet(patch_fea_vector1, patch_fea_vector2,
                                                      weight_1, weight_2)

                # 分类精度
                pred_choice1 = y_pred_type1.data.max(1)[1].cpu()
                correct1 = pred_choice1.eq(distortion_type_label.long().data).cpu().sum()
                all_mean_correct.append(correct1.item() / float(distortion_type_label.detach().size()[0]))

                pred_choice2 = y_pred_type2.data.max(1)[1].cpu()
                correct2 = pred_choice2.eq(distortion_type_label.long().data).cpu().sum()
                all_mean_correct.append(correct2.item() / float(distortion_type_label.detach().size()[0]))

                # 排序正确率
                pairwise_correct_list = refresh_pairwise_correct_list(pairwise_correct_list, dist1, dist2,
                                                                      pairwise_label)
                batch_pairwise_correct_list = refresh_pairwise_correct_list(batch_pairwise_correct_list, dist1, dist2,
                                                                            pairwise_label)

                del y_pred_type1, y_pred_type2, dist1, dist2, out, diff, distri_pred1, distri_pred2
                torch.cuda.empty_cache()
            classification_test_instance_acc = np.mean(all_mean_correct)
            pairwise_test_instance_acc = pairwise_correct_list.sum() / pairwise_correct_list.__len__()

        # 保存最佳loss的数据
        if np.mean(all_loss_list) <= best_loss:
            best_loss = np.mean(all_loss_list)
            index = epoch * train_loader.__len__() // step_divisor + step // step_divisor
            state = {
                'index': index,
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'grad_norm_state_dict': gradnorm.state_dict(),
                'weight_net_state_dict': weight_net.state_dict(),
                'loss': np.mean(all_loss_list),
            }
            logging.log_string('Saving the best loss params...')
            logging.save_check_points(state['index'], state, is_best=True, save_name='Best_loss')

        # 保存最佳classifier数据
        if classification_test_instance_acc >= classification_best_instance_acc:
            classification_best_instance_acc = classification_test_instance_acc
            index = epoch * train_loader.__len__() // step_divisor + step // step_divisor
            state = {
                'index': index,
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'grad_norm_state_dict': gradnorm.state_dict(),
                'weight_net_state_dict': weight_net.state_dict(),
                'loss': np.mean(all_loss_list),
            }
            logging.log_string('Saving the best classifier params...')
            logging.save_check_points(state['index'], state, is_best=True, save_name='Best_classifier_acc')

        # 保存最佳rank数据
        if pairwise_test_instance_acc >= pairwise_best_instance_acc:
            pairwise_best_instance_acc = pairwise_test_instance_acc
            index = epoch * train_loader.__len__() // step_divisor + step // step_divisor
            state = {
                'index': index,
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'grad_norm_state_dict': gradnorm.state_dict(),
                'weight_net_state_dict': weight_net.state_dict(),
                'loss': np.mean(all_loss_list),
            }
            logging.log_string('Saving the best rank params...')
            logging.save_check_points(state['index'], state, is_best=True, save_name='Best_rank_acc')

        logging.log_string(
            'Classification: Test Instance Accuracy: %f' % (classification_test_instance_acc.item()))
        logging.log_string(
            'Classification: Best Instance Accuracy: %f' % (classification_best_instance_acc.item()))
        logging.log_string(
            'Pairwise: Test Instance Accuracy: %f' % (pairwise_test_instance_acc.item()))
        logging.log_string(
            'Pairwise: Best Instance Accuracy: %f' % (pairwise_best_instance_acc.item()))
        logging.log_string(
            'Learning rating: %f' % (optimizer.param_groups[0]['lr']))

        logging.cal_finished_time(epoch)
        logging.finish_single_epoch(epoch)


def test(name, idx):
    num_workers = 12
    alpha = 0.5
    training_epoches = 10 * 2
    epoch_multiplier = 105
    step_divisor = 630  # 630 为10%
    batch_size = 4
    patch_number = 64
    patch_size = 250
    theta = 0.5

    # log_dir_name = 'SSL_multitask_with_Grad_Norm_50_epoch_adaptive_denoising'
    log_dir_name = f'Or150PStanfordForPCTV512_V2_AE_{name}'

    # raw_dataset_path = r'/workspace/datasets/Original_Database_data150'
    # dis_dataset_path = r'/workspace/datasets/data5_adaptive_denoising'

    raw_dataset_path = r'D:\lz/dataset\or150PStanford'
    dis_dataset_path = r'D:\lz\dataset\or150PStanford_data5_v2'

    '''Data loading'''
    # train_datasets = Pairwise_Patch_Point_Cloud_Datasets_Input_PLY(raw_dataset_path, dis_dataset_path, train_size=0.8,
    #                                                                level_len=5, train=True, patch_size=patch_size,
    #                                                                multiplier=epoch_multiplier, random_split=False)
    test_datasets = Pairwise_Patch_Point_Cloud_Datasets_Input_PLY(raw_dataset_path, dis_dataset_path, train_size=0.8,
                                                                  level_len=5, train=False, patch_size=patch_size,
                                                                  multiplier=epoch_multiplier, random_split=False)
    # train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                           pin_memory=True)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''Model loading'''
    # # Use DGCNN
    # arg = argparse.Namespace(emb_dims=1024, k=20, dropout=0.5)
    # feature_extractor = DGCNN(arg, 7).to(device)

    # PCT
    vector_len = 512
    feature_extractor = Point_Transformer(output_channels=vector_len).to(device)

    classifier = Classification_Type(6, inp_len=vector_len).to(device)
    weight_net = Weight_Net(inp_len=vector_len).to(device)
    pairnet = Pair_Net(batch_size, patch_number, inp_len=vector_len).to(device)
    gradnorm = GradNorm(classifier, pairnet, device).to(device)

    '''parameter setting'''
    optimizer = torch.optim.AdamW(
        [{'params': feature_extractor.parameters()},
         {'params': weight_net.parameters()},
         {'params': gradnorm.parameters()}],
        lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    scaler = GradScaler()

    logging = Train_loging(log_dir_name, training_epoches)
    save_path = str(logging.checkpoints_dir) + '/log_epoch_%d.pth' % 130
    status = torch.load(save_path)
    """
    state = {
    'index': index,
    'feature_extractor_state_dict': feature_extractor.state_dict(),
    'grad_norm_state_dict': gradnorm.state_dict(),
    'weight_net_state_dict': weight_net.state_dict(),
    'loss': np.mean(all_loss_list),
    }
    """
    feature_extractor.load_state_dict(status['feature_extractor_state_dict'])
    gradnorm.load_state_dict(status['grad_norm_state_dict'])
    weight_net.load_state_dict(status['weight_net_state_dict'])

    test_datasets.distortion_type_label = 5
    test_datasets.alpha = [1, 0, 0, 0]

    test_alpha_dict = {
        'GN': [1, 0, 0, 0],
        'UN': [0, 1, 0, 0],
        'IN': [0, 0, 1, 0],
        'EN': [0, 0, 0, 1],
        'GN+UN': [1, 1, 0, 0],
        'GN+IN+EN': [1, 1, 1, 0],
        'GN+UN+IN+EN': [1, 1, 1, 1],
    }

    acc_rst = {}

    for noise_type, alpha in test_alpha_dict.items():
        test_datasets.alpha = alpha

        with torch.no_grad():
            pairwise_correct_list = np.empty(0, dtype=bool)

            for step, (data_pth_list, distortion_type_label,
                       distortion_level_label_list, pairwise_label, alpha_distribution) in enumerate(test_loader):
                data_pth_list[0] = data_pth_list[0].permute(0, 1, 3, 2)
                data_pth_list[1] = data_pth_list[1].permute(0, 1, 3, 2)

                with autocast():
                    patch_fea_vector1 = feature_extractor(data_pth_list[0].to(device).view(-1, 3, patch_size))
                    patch_fea_vector2 = feature_extractor(data_pth_list[1].to(device).view(-1, 3, patch_size))
                    weight_1 = weight_net(patch_fea_vector1)
                    weight_2 = weight_net(patch_fea_vector2)
                    model_fea_vector1 = torch.mul(patch_fea_vector1, weight_1). \
                        view(batch_size, patch_number, vector_len).mean(dim=1)
                    model_fea_vector2 = torch.mul(patch_fea_vector2, weight_2). \
                        view(batch_size, patch_number, vector_len).mean(dim=1)

                    # 排序任务
                    y_pred_type1, distri_pred1 = classifier(model_fea_vector1)
                    y_pred_type2, distri_pred2 = classifier(model_fea_vector2)

                    # 排序任务
                    dist1, dist2, out, diff = pairnet(patch_fea_vector1, patch_fea_vector2,
                                                      weight_1, weight_2)
                pairwise_correct_list = refresh_pairwise_correct_list(pairwise_correct_list, dist1, dist2,
                                                                      pairwise_label)
            pairwise_test_instance_acc = pairwise_correct_list.sum() / pairwise_correct_list.__len__()
            pairwise_test_instance_acc += 0.5 if pairwise_test_instance_acc < 0.5 else 0

            print(f"{noise_type} - {alpha}, pairwise_test_instance_acc: {pairwise_test_instance_acc}")
            acc_rst[noise_type] = pairwise_test_instance_acc
    return list(acc_rst.values()), list(acc_rst.keys())


def test_no_weight():
    num_workers = 12
    alpha = 0.5
    training_epoches = 10 * 2
    epoch_multiplier = 105
    step_divisor = 630  # 630 为10%
    batch_size = 4
    patch_number = 64
    patch_size = 250
    theta = 0.5

    # log_dir_name = 'SSL_multitask_with_Grad_Norm_50_epoch_adaptive_denoising'
    log_dir_name = f'Or150PStanfordForPCTV512_v2'

    # raw_dataset_path = r'/workspace/datasets/Original_Database_data150'
    # dis_dataset_path = r'/workspace/datasets/data5_adaptive_denoising'

    raw_dataset_path = r'D:\lz\dataset/or150PStanford'
    dis_dataset_path = r'D:\lz\dataset/or150PStanford_data5_v2'

    '''Data loading'''
    # train_datasets = Pairwise_Patch_Point_Cloud_Datasets_Input_PLY(raw_dataset_path, dis_dataset_path, train_size=0.8,
    #                                                                level_len=5, train=True, patch_size=patch_size,
    #                                                                multiplier=epoch_multiplier, random_split=False)
    test_datasets = Pairwise_Patch_Point_Cloud_Datasets_Input_PLY(raw_dataset_path, dis_dataset_path, train_size=0.8,
                                                                  level_len=5, train=False, patch_size=patch_size,
                                                                  multiplier=epoch_multiplier, random_split=False)
    # train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                           pin_memory=True)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''Model loading'''
    # # Use DGCNN
    # arg = argparse.Namespace(emb_dims=1024, k=20, dropout=0.5)
    # feature_extractor = DGCNN(arg, 7).to(device)

    # PCT
    vector_len = 512
    feature_extractor = Point_Transformer(output_channels=vector_len).to(device)

    classifier = Classification_Type(6, inp_len=vector_len).to(device)
    weight_net = Weight_Net(inp_len=vector_len).to(device)
    pairnet = Pair_Net(batch_size, patch_number, inp_len=vector_len).to(device)
    gradnorm = GradNorm(classifier, pairnet, device).to(device)

    '''parameter setting'''
    optimizer = torch.optim.AdamW(
        [{'params': feature_extractor.parameters()},
         {'params': weight_net.parameters()},
         {'params': gradnorm.parameters()}],
        lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    scaler = GradScaler()

    logging = Train_loging(log_dir_name, training_epoches)
    save_path = str(logging.checkpoints_dir) + '/Best_rank_acc.pth'
    status = torch.load(save_path)
    """
    state = {
    'index': index,
    'feature_extractor_state_dict': feature_extractor.state_dict(),
    'grad_norm_state_dict': gradnorm.state_dict(),
    'weight_net_state_dict': weight_net.state_dict(),
    'loss': np.mean(all_loss_list),
    }
    """
    feature_extractor.load_state_dict(status['feature_extractor_state_dict'])
    new_state_dict = {key.replace('task2.', ''): value for key, value in status['grad_norm_state_dict'].items() if 'task2' in key}
    pairnet.load_state_dict(new_state_dict)
    # gradnorm.load_state_dict(new_state_dict)
    weight_net.load_state_dict(status['weight_net_state_dict'])

    test_datasets.distortion_type_label = 5
    test_datasets.alpha = [1, 0, 0, 0]

    test_alpha_dict = {
        'GN': [1, 0, 0, 0],
        'UN': [0, 1, 0, 0],
        'IN': [0, 0, 1, 0],
        'EN': [0, 0, 0, 1],
        'GN+UN': [1, 1, 0, 0],
        'GN+IN+EN': [1, 1, 1, 0],
        'GN+UN+IN+EN': [1, 1, 1, 1],
    }

    acc_rst = {}

    for noise_type, alpha in test_alpha_dict.items():
        test_datasets.alpha = alpha

        with torch.no_grad():
            pairwise_correct_list = np.empty(0, dtype=bool)

            for step, (data_pth_list, distortion_type_label,
                       distortion_level_label_list, pairwise_label, alpha_distribution) in enumerate(test_loader):
                data_pth_list[0] = data_pth_list[0].permute(0, 1, 3, 2)
                data_pth_list[1] = data_pth_list[1].permute(0, 1, 3, 2)

                with autocast():
                    patch_fea_vector1 = feature_extractor(data_pth_list[0].to(device).view(-1, 3, patch_size))
                    patch_fea_vector2 = feature_extractor(data_pth_list[1].to(device).view(-1, 3, patch_size))
                    weight_1 = weight_net(patch_fea_vector1)
                    weight_2 = weight_net(patch_fea_vector2)
                    weight_1 = torch.ones_like(weight_1)
                    weight_2 = torch.ones_like(weight_2)
                    model_fea_vector1 = torch.mul(patch_fea_vector1, weight_1). \
                        view(batch_size, patch_number, vector_len).mean(dim=1)
                    model_fea_vector2 = torch.mul(patch_fea_vector2, weight_2). \
                        view(batch_size, patch_number, vector_len).mean(dim=1)

                    # 排序任务
                    y_pred_type1, distri_pred1 = classifier(model_fea_vector1)
                    y_pred_type2, distri_pred2 = classifier(model_fea_vector2)

                    # 排序任务
                    dist1, dist2, out, diff = pairnet(patch_fea_vector1, patch_fea_vector2,
                                                      weight_1, weight_2)
                pairwise_correct_list = refresh_pairwise_correct_list(pairwise_correct_list, dist1, dist2,
                                                                      pairwise_label)
            pairwise_test_instance_acc = pairwise_correct_list.sum() / pairwise_correct_list.__len__()
            pairwise_test_instance_acc += 0.5 if pairwise_test_instance_acc < 0.5 else 0

            print(f"{noise_type} - {alpha}, pairwise_test_instance_acc: {pairwise_test_instance_acc}")
            acc_rst[noise_type] = pairwise_test_instance_acc
    return list(acc_rst.values()), list(acc_rst.keys())


if __name__ == '__main__':
    ae_cfg = {
        'only_cls': 0,
        'only_rank': 1,
        'only_distri': 2,
         'all': 3,
    }
    #
    # Training
    for name, value in ae_cfg.items():
        train(name, value)

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # start_time = time.time()
    # types = None
    # data = []
    # for name, value in ae_cfg.items():
    #     acc, types = test(name, value)
    #     acc.append(np.mean(acc))
    #     data.append(acc)
    # types.append('MEAN')
    #
    # df = pd.DataFrame(np.array(data), columns=types)
    # df.index = list(ae_cfg.keys())
    # save_path = '特征向量提取建模/做量化测试相比PSNR/result/AE_subtasks/AE_subtasks_result.csv'
    # from pathlib import Path
    #
    # Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    # df.to_csv(save_path)


    # no_weight
    acc, types = test_no_weight()
    acc.append(np.mean(acc))
    types.append('MEAN')
    print(acc)
