import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_distance.chamfer_loss import ChamferLoss


# SPD 的 loss
class Loss(nn.Module):
    """ Loss for training DeformNet.
        Use NOCS coords to supervise training.
    """
    def __init__(self, corr_wt, cd_wt, entropy_wt, deform_wt):
        super(Loss, self).__init__()
        self.threshold = 0.1
        self.chamferloss = ChamferLoss()
        self.corr_wt = corr_wt
        self.cd_wt = cd_wt
        self.entropy_wt = entropy_wt
        self.deform_wt = deform_wt

    # 问题来了, 之前求对应矩阵，是因为变形后的prior与gt点云nocs都是出于NOCS空间下，而且没有sRT的影响，因此两者理论上应该是重合的，
    # 而两帧的点云points_1和points_2是有点云影响的。
    # 难道要重建？
    def get_corr_loss(self, soft_assign_1, points_1, points_2):
        """
        Args:
            soft_assign_1: bs x n_pts x nv
            points_1: bs x nv x 3
            points_2: bs x nv x 3
        """
        # smooth L1 loss for correspondences
        # 对对应矩阵A的最后一维,缩放到(0,1)并使和为1
        # bs x n_pts x nv
        # 10,1024,1024
        # 使对应矩阵的每一行成为对应点的权重
        coords_1 = torch.bmm(soft_assign_1, points_1)  # (bs, n_pts, 3)
        diff = torch.abs(coords_1 - points_2)  # (bs, n_pts, 3)
        less = torch.pow(diff, 2) / (2.0 * self.threshold)
        higher = diff - self.threshold / 2.0
        corr_loss = torch.where(diff > self.threshold, higher, less)  # (idx0, idx1)
        corr_loss = torch.mean(torch.sum(corr_loss, dim=2))  # 对每个对应矩阵的行求和,然后对每行的和求平均值
        corr_loss = self.corr_wt * corr_loss
        return corr_loss

    def forward(self, assign_mat_1, points_1, assign_mat_2, points_2):
        # def forward(self, assign_mat, deltas, prior, nocs, model):
        """
        Args:
            assign_mat: bs x n_pts x nv
            deltas: bs x nv x 3
            prior: bs x nv x 3
        """
        # SPD的loss包含四部分
        # 1. Correspondence Loss
        # 类先验施加形变场,得到恢复出的实例模型
        inst_shape = prior + deltas

        soft_assign_1 = F.softmax(assign_mat_1, dim=2)
        soft_assign_2 = F.softmax(assign_mat_2, dim=2)

        self.get_corr_loss(soft_assign_1, points_1, points_2)
        self.get_corr_loss(soft_assign_2, points_2, points_1)
        # soft_assign与重建的点云相乘,得到预测的nocs坐标
        # 与实际的nocs作差,得到误差diff
        # torch.bmm batch matrix multiply 两个参数的第一维度应该是相同的
        # 矩阵相乘 (bs, n_pts, nv) x (bs, nv, 3)
        coords_1 = torch.bmm(soft_assign_1, points_1)  # (bs, n_pts, 3)
        coords_2 = torch.bmm(soft_assign_2, points_2)  # (bs, n_pts, 3)
        # inst_shape和nocs点的数量都是n_pts(被choose约束了)
        # 变形后的prior(coords)与gt点云(nocs)的误差
        # smooth L1 loss
        diff = torch.abs(coords - nocs)  # (bs, n_pts, 3)
        less = torch.pow(diff, 2) / (2.0 * self.threshold)
        higher = diff - self.threshold / 2.0
        corr_loss = torch.where(diff > self.threshold, higher, less)  # (idx0, idx1)
        corr_loss = torch.mean(torch.sum(corr_loss, dim=2))  # 对每个对应矩阵的行求和,然后对每行的和求平均值
        corr_loss = self.corr_wt * corr_loss

        # 2. Regularization Loss
        # entropy loss to encourage peaked distribution
        log_assign = F.log_softmax(assign_mat, dim=2)
        entropy_loss = torch.mean(-torch.sum(soft_assign * log_assign, 2))
        entropy_loss = self.entropy_wt * entropy_loss

        # 3. CD Loss
        # cd-loss for instance reconstruction
        cd_loss, _, _ = self.chamferloss(inst_shape, model)
        cd_loss = self.cd_wt * cd_loss

        # 4. Deform Loss
        # L2 regularizations on deformation
        # deform_loss = torch.norm(deltas, p=2, dim=2).mean()
        # deform_loss = self.deform_wt * deform_loss
        # total loss
        total_loss = corr_loss + entropy_loss + cd_loss + deform_loss
        return total_loss, corr_loss, cd_loss, entropy_loss, deform_loss
