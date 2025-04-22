# -*- coding: utf-8 -*-
"""
@author lizheng
@date  1:07
@packageName
@className GradNorm
@software PyCharm
@version 1.0.0
@describe TODO
"""

from torch import nn
import torch
import torch.nn.functional as F


class GradNorm(nn.Module):
    def __init__(self, task1_net, task2_net, device):
        super(GradNorm, self).__init__()
        self.task1 = task1_net
        self.task2 = task2_net
        self.device = device

        self.weights = torch.nn.Parameter(torch.ones(3).float())

    def forward(self, patch_fea_vector1, patch_fea_vector2, weight_1, weight_2,
                model_fea_vector1, model_fea_vector2, distortion_type_label, pairwise_label, alpha_distribution):
        # 获取分类任务
        cls_pred_type1, distri_pred1 = self.task1(model_fea_vector1)
        cls_pred_type2, distri_pred2 = self.task1(model_fea_vector2)

        # 排序任务
        dist1, dist2, out, diff = self.task2(patch_fea_vector1, patch_fea_vector2, weight_1, weight_2)

        # 求取分类和排序的损失
        # breakpoint()
        loss1 = F.cross_entropy(cls_pred_type1, distortion_type_label.to(self.device))
        loss2 = F.cross_entropy(cls_pred_type2, distortion_type_label.to(self.device))
        loss1_kl = F.kl_div(distri_pred1.log(), alpha_distribution, reduction='batchmean')
        loss2_kl = F.kl_div(distri_pred2.log(), alpha_distribution, reduction='batchmean')
        loss_classification = (loss1 + loss2).unsqueeze(dim=0)
        distri_loss = (loss1_kl + loss2_kl).unsqueeze(dim=0)
        loss_rank = F.binary_cross_entropy_with_logits(diff, pairwise_label.float().to(self.device)).unsqueeze(dim=0)
        task_loss = torch.concatenate([loss_classification, loss_rank, distri_loss])

        if torch.isnan(loss1) or torch.isnan(loss2):
            print("nan")
            breakpoint()

        return task_loss, cls_pred_type1, cls_pred_type2, dist1, dist2, distri_pred1, distri_pred2


if __name__ == '__main__':
    ...
