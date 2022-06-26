import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_distance.chamfer_loss import ChamferLoss
from network.lib.utils import render_points_diff_color, render_lines
from captra_utils.utils_from_captra import project

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
    def get_corr_loss_past(self, soft_assign_1, points_1, points_2, sRt):
        """
                Args:
                    soft_assign_1: bs x n_pts x nv
                    points_1: bs x nv x 3
                    points_2: bs x nv x 3
                """
        soft_assign_1 = soft_assign_1.type(torch.float64)
        points_1 = points_1.type(torch.float64)
        points_2 = points_2.type(torch.float64)
        # pose12_gt['scale'] = pose12_gt['scale'].type(torch.float64)
        # pose12_gt['rotation'] = pose12_gt['rotation'].type(torch.float64)
        # pose12_gt['translation'] = pose12_gt['translation'].type(torch.float64)
        sRt = sRt.type(torch.float64)
        # smooth L1 loss for correspondences
        # 对对应矩阵A的最后一维,缩放到(0,1)并使和为1
        # bs x n_pts x nv
        # 10,1024,1024
        # 使对应矩阵的每一行成为对应点的权重
        points_1_in_2 = torch.bmm(soft_assign_1, points_2)  # (bs, n_pts, 3) points_1_in_2为points_1在points_2坐标系下的映射
        # 计算diff的时候得想办法
        # 1. 根据coord图的对应关系，只有有对应关系的点才相减
        # 2. 根据gt RT 将points_1变换到points_2，求两者对应关系
        # 注意【nocs与points是一一对应的，可以直接相减】
        # 可以比较一下这两种方法计算出的loss的值
        # diff = torch.abs(points_1_in_2 - points_2)  # (bs, n_pts, 3)

        # 使用pose12_gt将point1变换到point2位置处
        # 然后计算diff
        # pts1to2_R = torch.bmm(pose12_gt['rotation'], points_1.transpose(-2, -1))  # (bs,3,3)x(bs,3,1024)=(bs,3,1024)
        # pts1to2_sR = pts1to2_R * pose12_gt['scale'].view((-1, 1, 1))
        # pts1to2_sRt = pts1to2_sR + pose12_gt['translation'].transpose(1, 2).squeeze(-1)
        # pts1to2_sRt = pts1to2_sRt.transpose(-1, -2)

        # gt
        points_1to2_newMethod = self.multi_pose_with_pts(sRt, points_1)
        # problem: diff是否满足位移不变性？是否应该除以尺寸？
        # 是否应该用ChamferDistance代替直接做差？不，CD无法评估对应点的误差
        diff = torch.abs(points_1_in_2 - points_1to2_newMethod)  # (bs, n_pts, 3)

        less = torch.pow(diff, 2) / (2.0 * self.threshold)
        higher = diff - self.threshold / 2.0
        corr_loss = torch.where(diff > self.threshold, higher, less)  # (idx0, idx1)
        corr_loss = torch.mean(torch.sum(corr_loss, dim=2))  # 对每个对应矩阵的行求和,然后对每行的和求平均值
        corr_loss = self.corr_wt * corr_loss

        return corr_loss

    def get_corr_loss(self, points_1_in_2, points_1to2_gt):
        """
                Args:
                    soft_assign_1: bs x n_pts x nv
                    points_1: bs x nv x 3
                    points_2: bs x nv x 3
                """
        # problem: diff是否满足位移不变性？是否应该除以尺寸？
        # 是否应该用ChamferDistance代替直接做差？不，CD无法评估对应点的误差
        diff = torch.abs(points_1_in_2 - points_1to2_gt)  # (bs, n_pts, 3)

        less = torch.pow(diff, 2) / (2.0 * self.threshold)
        higher = diff - self.threshold / 2.0
        # smooth L1 loss对于大于阈值的点
        corr_loss = torch.where(diff > self.threshold, higher, less)  # (idx0, idx1)
        corr_loss = torch.mean(torch.sum(corr_loss, dim=2))  # 对每个对应矩阵的行求和,然后对每行的和求平均值
        corr_loss = self.corr_wt * corr_loss

        print("smooth L1 loss: num of points diff > threshold {0}".format(len(torch.where(diff[0] > 0.1)[0])))

        return corr_loss

    def get_RegularizationLoss(self, assign_mat, soft_assign):
        log_assign = F.log_softmax(assign_mat, dim=2)
        entropy_loss = torch.mean(-torch.sum(soft_assign * log_assign, 2))
        entropy_loss = self.entropy_wt * entropy_loss
        return entropy_loss

    def get_RegularizationLoss_v(self, assign_mat, soft_assign):
        log_assign = F.log_softmax(assign_mat, dim=1)
        entropy_loss = torch.mean(-torch.sum(soft_assign * log_assign, 1))
        entropy_loss = self.entropy_wt * entropy_loss
        return entropy_loss

    def get_cos_sim_loss(self, mat1, mat2):
        assign_matmul = torch.bmm(mat1, mat2)
        eye_bs = torch.eye(assign_matmul.shape[1]).unsqueeze(0).repeat(len(assign_matmul), 0, 0)
        # 余弦相似性,如果两个矩阵完全一致,相似性的值为1
        cos_sim_loss = 1 - F.cosine_similarity(torch.flatten(mat1, 1), torch.flatten(mat2, 1))
        return cos_sim_loss.mean().item()

    def pose_dict_RT(self, pose_dict):
        scale = pose_dict['scale']    # (bs, 1)
        RMatrix = pose_dict['rotation']  # (bs, 3, 3)
        tVec = pose_dict['translation']  # (bs, 1, 3, 1)

        # sRt_ = torch.zeros((scale.shape[0], 4, 4))
        # sRt_[:, :3, :3] = RMatrix
        # sRt_[:, :3, 3] = tVec.squeeze(1).squeeze(-1)
        # sRt_[:, 3, 3] = 1

        sR = RMatrix * scale.view((-1, 1, 1))  # (bs, 3, 3)
        sRt = torch.cat((sR, tVec.transpose(1, 2).squeeze(-1)), 2)  # (bs, 3, 4)
        bottom = torch.zeros(RMatrix.shape[0], 1, 4).cuda()  # (bs, 1, 4)
        bottom[:, :, 3] = 1
        sRt01 = torch.cat((sRt, bottom), 1)  # (bs, 4, 4)
        return sRt01

    def multi_pose_with_pts(self, pose_sRT, points):
        # points    (bs, 1024, 3)
        # pose_sRT  (bs, 4, 4)
        pts = points.clone()
        pts = pts.transpose(1, 2)  # (bs, 3, 1024)
        one_ = torch.ones(pts.shape[0], 1, 1024).cuda()  # (bs, 1, 1024) fill 1
        pts = torch.cat((pts, one_), 1)  # (bs, 4, 1024)
        res_pts_4 = torch.bmm(pose_sRT, pts)  # (bs, 4, 4)*(bs, 4, 1024)
        # (bs, 4, 1024)
        res_pts_4 = res_pts_4[:, 0:3, :].transpose(-1, -2)
        return res_pts_4

    def multi_pose_with_pts_debug(self, pose_sRT, points, debug):
        # points    (bs, 1024, 3)
        # pose_sRT  (bs, 4, 4)
        pts = points.clone()
        pts = pts.transpose(1, 2)  # (bs, 3, 1024)
        one_ = torch.ones(pts.shape[0], 1, 1024).cuda()  # (bs, 1, 1024) fill 1
        pts = torch.cat((pts, one_), 1)  # (bs, 4, 1024)
        pose_sRT_ = pose_sRT.clone()
        if debug == 1:
            pose_sRT_[:, :3, :3] = 0
            pose_sRT_[:, 0, 0] = 1
            pose_sRT_[:, 1, 1] = 1
            pose_sRT_[:, 2, 2] = 1
        elif debug == 2:
            pose_sRT_[:, :3, 3] = 0
        elif debug == 3:
            pose_sRT_[:, :3, :3] = 0
            pose_sRT_[:, 0, 0] = 1
            pose_sRT_[:, 1, 1] = 1
            pose_sRT_[:, 2, 2] = 1
            pose_sRT_[:, :3, 3] = 0
        print(f'pose_Rt_{debug}: \n{pose_sRT_}')
        res_pts_4 = torch.bmm(pose_sRT_, pts)  # (bs, 4, 4)*(bs, 4, 1024)
        # (bs, 4, 1024)
        res_pts_4 = res_pts_4[:, 0:3, :].transpose(-1, -2)
        return res_pts_4

    # 将pose反转
    def pose_inverse(self, sRt_bs):
        sRt_bs_inv = torch.inverse(sRt_bs)
        return sRt_bs_inv


    def get_total_loss_2_frame(self, points_1,  points_2, assign_mat_1, assign_mat_2, pose12_gt, m1m2):
        # def forward(self, assign_mat, deltas, prior, nocs, model):
        """
        Args:
            assign_mat: bs x n_pts x nv
            deltas: bs x nv x 3
            prior: bs x nv x 3
        """
        model_1, model_2, gt_model_numpy, model_1_12 = m1m2

        soft_assign_1 = F.softmax(assign_mat_1, dim=2)
        soft_assign_2 = F.softmax(assign_mat_2, dim=2)

        sRt12_gt = self.pose_dict_RT(pose12_gt)
        sRt21_gt = self.pose_inverse(sRt12_gt)  # 每个batch对应的两个矩阵是互逆的

        # 投影后点云与gt点云
        soft_assign_1 = soft_assign_1.type(torch.float64)
        soft_assign_2 = soft_assign_2.type(torch.float64)
        points_1 = points_1.type(torch.float64)
        points_2 = points_2.type(torch.float64)
        sRt12_gt = sRt12_gt.type(torch.float64)
        sRt21_gt = sRt21_gt.type(torch.float64)
        points_1_in_2 = torch.bmm(soft_assign_1, points_2)  # (bs, n_pts, 3) points_1_in_2为points_1在points_2坐标系下的映射
        points_2_in_1 = torch.bmm(soft_assign_2, points_1)
        # gt
        points_1to2_gt = self.multi_pose_with_pts(sRt12_gt, points_1)
        points_2to1_gt = self.multi_pose_with_pts(sRt21_gt, points_2)
        # debug
        points_1to2_RI = self.multi_pose_with_pts_debug(sRt12_gt, points_1, 1)
        points_1to2_tI = self.multi_pose_with_pts_debug(sRt12_gt, points_1, 2)
        points_1to2_RtI = self.multi_pose_with_pts_debug(sRt12_gt, points_1, 3)


        # 0. CD loss 能否约束形状？
        cd_loss1, _, _ = self.chamferloss(points_1to2_gt.type(torch.float32).contiguous(), points_1_in_2.type(torch.float32))
        cd_loss2, _, _ = self.chamferloss(points_2to1_gt.type(torch.float32).contiguous(), points_2_in_1.type(torch.float32))
        # 1. Correspondence Loss

        corr_loss_1 = self.get_corr_loss(points_1_in_2, points_1to2_gt)
        corr_loss_2 = self.get_corr_loss(points_2_in_1, points_2to1_gt)

        # 2. Regularization Loss
        # entropy loss to encourage peaked distribution
        entropy_loss_1 = self.get_RegularizationLoss(assign_mat_1, soft_assign_1)
        entropy_loss_2 = self.get_RegularizationLoss(assign_mat_2, soft_assign_2)

        entropy_loss_1v = self.get_RegularizationLoss_v(assign_mat_1, soft_assign_1)
        entropy_loss_2v = self.get_RegularizationLoss_v(assign_mat_2, soft_assign_2)

        # 3. A1和A2应当互逆
        reciprocal_loss = self.get_cos_sim_loss(assign_mat_1, assign_mat_2)

        # loss weight
        corr_loss_1 = 1.0 * corr_loss_1
        corr_loss_2 = 1.0 * corr_loss_2
        entropy_loss_1 = 1.0 * entropy_loss_1
        entropy_loss_2 = 1.0 * entropy_loss_2
        entropy_loss_1v = 1.0 * entropy_loss_1v
        entropy_loss_2v = 1.0 * entropy_loss_2v
        reciprocal_loss = 1.0 * reciprocal_loss

        total_loss = cd_loss1 + cd_loss2 + corr_loss_1 + corr_loss_2 + entropy_loss_1 + entropy_loss_2 + reciprocal_loss + entropy_loss_1v + entropy_loss_2v

        # if corr_loss_1 < 0.10:
        batch_idx = 0
        I_matrix = torch.eye(1024).unsqueeze(0).repeat(10, 1, 1)
        I_gt = torch.eye(4).unsqueeze(0).repeat(10, 1, 1)
        soft_assign_1 = soft_assign_1.type(torch.float64)
        points_1 = points_1.type(torch.float64)
        points_2 = points_2.type(torch.float64)
        points_1_in_2 = torch.bmm(soft_assign_1, points_2)
        assigned_points = points_1_in_2[batch_idx].cpu().detach().numpy()
        points_1 = points_1[batch_idx].cpu().detach().numpy()
        points_2 = points_2[batch_idx].cpu().detach().numpy()
        # RGB 颜色已验证没问题
        color_red = np.array([255, 0, 0])
        color_green = np.array([0, 255, 0])
        color_blue = np.array([0, 0, 255])
        color_blue2 = np.array([0, 0, 100])
        color_gray = np.array([93, 93, 93])
        color_black = np.array([255, 255, 255])
        pts_colors = [color_green, color_red]

        # 输入点云1,2，输出重新排序后的2使其与1点对应
        def sort_corr_points(p1, p2):
            # 在找points_1和points_2的对应关系
            p_sort = []
            ref = []
            for i in range(p1.shape[0]):
                xyz = p1[i]
                tmp = torch.abs(p2 - xyz.unsqueeze(0).repeat(p2.shape[0], 1))
                tmp = torch.sum(tmp, 1)
                idx = torch.argmin(tmp)
                p_sort.append(p2[idx])
                ref.append(idx)
            # 合并重排序后的点云，可能会有重复的点(有的点可能没有被选择)
            p2_resorted = torch.stack(p_sort, dim=0)
            return ref, p2_resorted

        mean1 = np.mean(points_1, 0).reshape(1, 3)
        mean2 = np.mean(points_2, 0).reshape(1, 3)

        render_points_diff_color('model 1, 2, 1_12',
                                 [model_1, model_2, model_1_12],
                                 [color_red, color_green, color_blue],
                                 save_img=False,
                                 show_img=True)

        render_points_diff_color('model and pts 1',
                                 [points_1, model_1, gt_model_numpy],
                                 [color_red, color_green, color_blue],
                                 save_img=False,
                                 show_img=True)

        render_points_diff_color('model and pts 2',
                                 [points_2, model_2, gt_model_numpy],
                                 [color_red, color_green, color_blue],
                                 save_img=False,
                                 show_img=True)

        # 投影到图像上


        render_points_diff_color('2',
                                 [points_2to1_gt[batch_idx].cpu().numpy(),
                                  points_1to2_gt[batch_idx].cpu().numpy(),
                                  points_1,
                                  points_2,
                                  points_1to2_RI[batch_idx].cpu().numpy(),
                                  points_1to2_tI[batch_idx].cpu().numpy()],
                                 [color_green, color_red, color_blue, color_blue2, color_black, color_gray], save_img=False,
                                 show_img=True)


        render_points_diff_color('3',
                                  [points_1,
                                  points_1to2_RtI[batch_idx].cpu().numpy(),
                                  points_1to2_RI[batch_idx].cpu().numpy(),
                                  points_1to2_tI[batch_idx].cpu().numpy()],
                                 [color_green, color_red, color_blue, color_gray],
                                 save_img=False,
                                 show_img=True)

        ref, p2_rs = sort_corr_points(points_1to2_gt[0], torch.from_numpy(points_2).cuda())  # 返回的是两个tensor

        lines = []
        offset = len(ref)
        points_draw = np.vstack((points_1to2_gt[0].cpu().numpy(), p2_rs.cpu().numpy()))
        for i in range(len(ref)):
            lines.append([i, ref[i].item() + offset])

        render_lines("sort corr", [points_1to2_gt[0].cpu().numpy(), points_2], pts_colors, points_draw, lines)

        render_points_diff_color('points_1:green points_2:red', [points_1, points_2],
                                 pts_colors, save_img=False,
                                 show_img=True)
        render_points_diff_color('points_1 in 2:green points_2:red', [assigned_points, points_2],
                                 pts_colors, save_img=False,
                                 show_img=True)
        print(soft_assign_1)
        print(soft_assign_1)

        return total_loss, cd_loss1, cd_loss2, corr_loss_1, corr_loss_2, entropy_loss_1, entropy_loss_2, reciprocal_loss, entropy_loss_1v, entropy_loss_2v

    def forward(self, points_assign_mat_list, pose12_gt, m1m2):
        total_loss = 0.0
        for frame_pair in points_assign_mat_list:
            points_bs_1, points_bs_2, assign_matrix_bs_1, assign_matrix_bs_2 = frame_pair
            frame_total_loss, cd_loss1, cd_loss2, corr_loss_1, corr_loss_2, entropy_loss_1, entropy_loss_2, reciprocal_loss, entropy_loss_1v, entropy_loss_2v = \
                self.get_total_loss_2_frame(points_bs_1, points_bs_2, assign_matrix_bs_1, assign_matrix_bs_2, pose12_gt, m1m2)
            print("cd_loss_1:      {7},\n"
                  "        2       {8}\n"
                  "corr_loss_1:    {0},\n"
                  "          2     {1}\n"
                  "entropy_loss:   {2},\n"
                  "                {3}\n"
                  "entropy_l  v:   {5},\n"
                  "                {6}\n"
                  "reciprocal_loss:{4}\n".format(corr_loss_1, corr_loss_2, entropy_loss_1, entropy_loss_2, reciprocal_loss, entropy_loss_1v, entropy_loss_2v, cd_loss1, cd_loss2))
            total_loss += 1.0 * frame_total_loss
        return total_loss

