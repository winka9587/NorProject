import torch
import torch.nn as nn
import torch.nn.functional as F
from part_dof_utils import part_model_batch_to_part, eval_part_full, add_noise_to_part_dof, \
    compute_parts_delta_pose
from utils import cvt_torch, Timer, add_border_bool_by_crop_pos
from network.lib.utils import crop_img
import numpy as np
from extract_2D_kp import extract_sift_kp_from_RGB, sift_match
import cv2
from normalspeedTest import norm2bgr
# network.FFB6D_models.
# from network.FFB6D_models.cnn.pspnet import PSPNet
import network.FFB6D_models.pytorch_utils as pt_utils
# from network.FFB6D_models.RandLA.RandLANet import Network as RandLANet
from captra_utils.utils_from_captra import backproject

from lib.pspnet import PSPNet
from lib.pointnet import Pointnet2MSG
from lib.loss import Loss

from torch.optim import lr_scheduler

from visualize import viz_multi_points_diff_color, viz_mask_bool

class ConfigRandLA:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 480 * 640 // 24  # Number of input points
    num_classes = 22  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 3  # batch_size during training
    val_batch_size = 3  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch
    in_c = 9

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [32, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]


# psp_models = {
#     'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
#     'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
#     'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
# }
def get_scheduler(optimizer, opt, it=-1):
    scheduler = None
    if optimizer is None:
        return scheduler
    if opt.lr_policy is None or opt.lr_policy == 'constant':
        scheduler = None  # constant scheduler
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_step_size,
                                        gamma=opt.lr_gamma,
                                        last_epoch=it)
    else:
        assert 0, '{} not implemented'.format(opt.lr_policy)
    return scheduler


def get_optimizer(params, opt):
    if len(params) == 0:
        return None
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params, lr=opt.learning_rate,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=opt.weight_decay)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=opt.learning_rate,
            momentum=0.9)
    else:
        assert 0, "Unsupported optimizer type {}".format(opt.optimizer)
    return optimizer


class SIFT_Track(nn.Module):
    def __init__(self, device, real, subseq_len=2, mode='train', opt=None):
        super(SIFT_Track, self).__init__()
        # self.fc1 = nn.Linear(emb_dim, 512)
        # self.fc2 = nn.Linear(512, 1024)
        # self.fc3 = nn.Linear(1024, 3*n_pts)
        self.mode = mode
        self.subseq_len = subseq_len
        self.device = device
        self.num_parts = 1
        self.num_joints = 0
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        if real:
            self.intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
            self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = 591.0125, 590.16775, 322.525, 244.11084
        else:
            self.intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])
            self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = 577.5, 577.5, 319.5, 239.5

        # SGPA
        self.psp = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')
        self.n_pts = 1024  # 从mask中采样1024个点，提取颜色特征
        self.instance_geometry = Pointnet2MSG(0)
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.instance_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.assignment = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, self.n_pts, 1),
        )
        # Loss
        corr_wt = 1.0
        cd_wt = 5.0
        entropy_wt = 0.0001
        deform_wt = 0.01
        self.criterion = Loss(corr_wt, cd_wt, entropy_wt, deform_wt)  # SPD 的loss

        # 反投影用
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.norm_scale = 1000.0
        # FFB6D
        # cnn = psp_models['resnet34'.lower()]()
        # self.cnn_pre_stages = nn.Sequential(
        #     cnn.feats.conv1,  # stride = 2, [bs, c, 240, 320]
        #     cnn.feats.bn1, cnn.feats.relu,
        #     cnn.feats.maxpool  # stride = 2, [bs, 64, 120, 160]
        # )
        # rndla_cfg = ConfigRandLA
        # rndla = RandLANet(rndla_cfg)
        # self.rndla_pre_stages = rndla.fc0
        self.rndla_pre_stages = pt_utils.Conv1d(9, 8, kernel_size=1, bn=True)

        # ####################### downsample stages#######################
        # self.cnn_ds_stages = nn.ModuleList([
        #     cnn.feats.layer1,  # stride = 1, [bs, 64, 120, 160]
        #     cnn.feats.layer2,  # stride = 2, [bs, 128, 60, 80]
        #     # stride = 1, [bs, 128, 60, 80]
        #     nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),
        #     nn.Sequential(cnn.psp, cnn.drop_1)  # [bs, 1024, 60, 80]
        # ])
        # self.ds_sr = [4, 8, 8, 8]

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.learning_rate)
        self.epoch = 0
        self.loss_dict = {}


    # frame={dict:4}
    # points, labels, nocs, meta
    # 最终返回的feed_frame(也就是初始帧)包含
    # points, labels, meta, nocs, gt_part(nocs2camera的gt位姿)
    def convert_init_frame_data(self, frame):
        feed_frame = {}
        for key, item in frame.items():
            if key not in ['meta', 'labels', 'points', 'nocs']:
                continue
            if key in ['meta']:
                pass
            elif key in ['labels']:
                item = item.long().to(self.device)
            else:
                item = item.float().to(self.device)
            feed_frame[key] = item
        gt_part = part_model_batch_to_part(cvt_torch(frame['meta']['nocs2camera'], self.device), self.num_parts,
                                           self.device)
        feed_frame.update({'gt_part': gt_part})

        return feed_frame


    # 最终返回的input包含:(带*是相比于init frame多出来的)
    # points    (B,3,4096)
    # meta      (B,3,1)
    # labels    (B,4096)
    # gt_part   (rotation,scale,translation)
    # points_mean *     (10,3,1)
    # npcs *            (10,3,4096)
    def convert_subseq_frame_data(self, data):
        # 计算gt位姿
        gt_part = part_model_batch_to_part(cvt_torch(data['meta']['nocs2camera'], self.device), self.num_parts,
                                           self.device)
        # 观测点云,观测点云平均点,gt位姿
        input = {'points': data['points'],
                 'points_mean': data['meta']['points_mean'],
                 'gt_part': gt_part}
        # 添加nocs点云
        if 'nocs' in data:
            input['npcs'] = data['nocs']
        input = cvt_torch(input, self.device)
        # 添加meta
        input['meta'] = data['meta']
        # 添加labels
        if 'labels' in data:
            input['labels'] = data['labels'].long().to(self.device)
        return input

    # 最终的input包含:
    # points
    # nocs
    # labels
    # meta
    # points_mean
    def convert_subseq_frame_npcs_data(self, data):
        input = {}
        for key, item in data.items():
            if key not in ['meta', 'labels', 'points', 'nocs']:
                continue
            elif key in ['meta']:
                pass
            elif key in ['labels']:
                item = item.long().to(self.device)
            else:
                item = item.float().to(self.device)
            input[key] = item
        input['points_mean'] = data['meta']['points_mean'].float().to(self.device)
        return input

    # 估计对应矩阵
    # bs x n_pts x nv
    # (10, 1024, 1024)
    def get_assgin_matrix(self, inst_local, inst_global, cat_global):
        assign_feat_bs = torch.cat((inst_local, inst_global.repeat(1, 1, self.n_pts), cat_global.repeat(1, 1, self.n_pts)),
                                dim=1)  # bs x 2176 x n_pts
        assign_mat_bs = self.assignment(assign_feat_bs)  # bs x 1024 x 1024  (bs, n_pts_2, npts_1)
        # nv原本是prior点的数量,但这里因为用的是实例点云(只是不同帧而已),所以与self.n_pts相同
        nv = self.n_pts
        assign_mat_bs = assign_mat_bs.view(-1, nv, self.n_pts).contiguous()  # bs, nv, n_pts -> bs, nv, n_pts

        # assign_mat = torch.index_select(assign_mat, 0, index)  # 删除softmax这一部分
        assign_mat_bs = assign_mat_bs.permute(0, 2, 1).contiguous()  # bs x n_pts x nv
        return assign_mat_bs

    # 有mask_add_from_last_frame说明是第二帧，需要使用前一帧提供的mask
    def extract_3D_kp(self, frame, mask_bs):
        timer = Timer(True)
        # rgb                彩色图像               [bs, 3, h, w]
        # dpt_nrm            图像:xyz+normal       [bs, 6, h, w], 3c xyz in meter + 3c normal map
        # cld_rgb_nrm点云:    xyz+rgb+normal      [bs, 9, npts]
        # choose             应该是mask            [bs, 1, npts]

        # 输入color,normal,depth,mask,反投影，得到9通道的点云
        color_bs = frame['meta']['pre_fetched']['color']
        depth_bs = frame['meta']['pre_fetched']['depth']
        nrm_bs = frame['meta']['pre_fetched']['nrm']
        # 从mask中提取n_pts个点的下标 choose
        def choose_from_mask_bs(mask_bs):
            output_choose = torch.tensor([])
            mask_bs_choose = torch.full(mask_bs.shape, False)
            for b_i in range(len(mask_bs)):
                # get choose pixel index by mask
                mask = mask_bs[b_i]
                mask_flatten = torch.full(mask.flatten().shape, False)  # 为了确定采样点的像素坐标
                choose = torch.where(mask.flatten())[0]
                if len(choose) > self.n_pts:
                    choose_mask = np.zeros(len(choose), dtype=np.uint8)
                    choose_mask[:self.n_pts] = 1
                    np.random.shuffle(choose_mask)
                    choose_mask = torch.from_numpy(choose_mask)
                    choose = choose[choose_mask.nonzero()]
                else:
                    # 点的数量不够,需要补齐n_pts个点
                    # wrap 用前面补后面，用后面补前面
                    choose = torch.from_numpy(np.pad(choose.numpy(), (0, self.n_pts - len(choose)), 'wrap'))
                mask_flatten[choose] = True
                mask_flatten = mask_flatten.view(mask.shape).numpy()
                # 可视化mask
                # viz_mask_bool('mask_flatten', mask_flatten)
                choose = torch.unsqueeze(choose, 0)  # (n_pts, ) -> (1, n_pts)
                output_choose = torch.cat((output_choose, choose), 0)
                output_choose = output_choose.type(torch.int64)
            return output_choose
        choose_bs = choose_from_mask_bs(mask_bs)  # choose_bs (bs, n_pts=1024)
        choose_bs = choose_bs.cuda()
        # mask_bs_choose = mask_bs_choose.cuda()

        timer.tick('extract idx from mask')

        # 遍历batch
        bs = len(depth_bs)
        mask_bs_next = mask_bs.clone()
        color_bs_zero_pad = color_bs.clone()
        points_feat_bs = torch.tensor([]).cuda()  # 提取的几何特征
        points_bs = torch.tensor([]).cuda()  # (bs, 1024, 3)
        timer_1 = Timer(True)
        for batch_idx in range(bs):
            color = color_bs[batch_idx]
            depth = depth_bs[batch_idx]
            nrm = nrm_bs[batch_idx]
            mask = mask_bs[batch_idx]
            choose = choose_bs[batch_idx]

            # 1.提取几何特征
            # 根据mask,反投影得到观测点云
            # points, idxs = backproject(depth, self.intrinsics, mask=mask_bs[batch_idx])
            # CAPTRA 反投影得到点云
            # points, idxs = backproject(depth, self.intrinsics)
            # points = torch.from_numpy(points).cuda()
            # points_rgb = color[idxs[0], idxs[1]].cuda()
            # points_nrm = nrm[idxs[0], idxs[1]].cuda()
            # pts_6d = torch.concat([points, points_rgb], dim=1)
            # SPD反投影得到点云
            choose = choose.cpu().numpy()
            depth_masked = depth.flatten()[choose][:, np.newaxis]       # 对点云进行一个采样,采样n_pts个点
            xmap_masked = self.xmap.flatten()[choose][:, np.newaxis]     # 像素坐标u
            ymap_masked = self.ymap.flatten()[choose][:, np.newaxis]     # 像素坐标v
            pt2 = depth_masked / self.norm_scale
            pt2 = pt2.numpy()                               # z
            pt0 = (xmap_masked - self.cam_cx) * pt2 / self.cam_fx     # x
            pt1 = (ymap_masked - self.cam_cy) * pt2 / self.cam_fy     # y
            points = np.concatenate((pt0, pt1, pt2), axis=1)  # xyz
            points = torch.from_numpy(np.transpose(points, (2, 0, 1))).cuda()  # points (1024, 3, 1) -> (1, 1024, 3)
            points = points.type(torch.float32)
            points_bs = torch.cat((points_bs, points), 0)
            timer_1.tick('single batch | backproject')
            # 测试读取rgb
            # rgb_show = torch.zeros(color.shape, dtype=torch.uint8)
            # rgb_show[ymap_masked, xmap_masked, 0] = color[ymap_masked, xmap_masked, 0]
            # rgb_show[ymap_masked, xmap_masked, 1] = color[ymap_masked, xmap_masked, 1]
            # rgb_show[ymap_masked, xmap_masked, 2] = color[ymap_masked, xmap_masked, 2]
            # cv2.imshow('rgb', rgb_show.numpy())
            # cv2.waitKey(0)

            points_rgb = color[ymap_masked, xmap_masked]
            points_rgb = points_rgb.squeeze(1).transpose(1, 0).cuda()  # points_rgb (1024, 1, 3) -> (1, 1024, 3)
            points_nrm = nrm[ymap_masked, xmap_masked]
            points_nrm = points_nrm.squeeze(1).transpose(1, 0).cuda()  # points_nrm (1024, 1, 3) -> (1, 1024, 3)
            timer_1.tick('single batch | extract color and nrm feature')
            # 如果要增加对噪声点的过滤，需要重新采样(重新计算choose)，或者在计算choose之前就进行过滤
            # viz_multi_points_diff_color(f'pts:{batch_idx}', [points], [np.array([[1, 0, 0]])])
            # 拼接三个不同的特征
            # 是否可以留着nrm在损失函数里使用？
            #  -> (1, 1024, 3) ->  -> (1, 1024, 9)
            pts_9d = torch.concat([points, points_rgb, points_nrm], dim=2)
            points_feat_bs = torch.cat((points_feat_bs, pts_9d), 0)
            timer_1.tick('single batch | concat 3 feat')
            # 2.提取颜色特征
            # 通过mask获得crop_pos
            def get_crop_pos_by_mask(mask):
                crop_idx = torch.where(mask)
                x_max = torch.max(crop_idx[1]).item()
                x_min = torch.min(crop_idx[1]).item()
                y_max = torch.max(crop_idx[0]).item()
                y_min = torch.min(crop_idx[0]).item()
                crop_pos_tmp = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
                return crop_pos_tmp

            def zero_padding_by_mask(img, mask):
                t2 = Timer(True)
                crop_pos = get_crop_pos_by_mask(mask)
                t2.tick('--get crop by mask')
                img_zero_pad = torch.zeros(img.shape, dtype=img.dtype)
                # 创建idx[0] y
                # 创建indx[1] x

                idx1 = torch.arange(crop_pos['x_min'], crop_pos['x_max'])
                idx0 = torch.full(idx1.shape, crop_pos['y_min'])
                t2.tick('--fill zero')
                mask_crop_idx0 = idx0
                mask_crop_idx1 = idx1
                for y in range(1, crop_pos['y_max']-crop_pos['y_min']):
                    idx0 = torch.full(idx1.shape, crop_pos['y_min']+y)
                    mask_crop_idx0 = torch.concat((mask_crop_idx0, idx0))
                    mask_crop_idx1 = torch.concat((mask_crop_idx1, idx1))
                mask_crop_idx = (mask_crop_idx0, mask_crop_idx1)
                img_zero_pad[mask_crop_idx] = img[mask_crop_idx]
                t2.tick('--map mask value to img')
                # 可视化零填充后的mask,bbox之外应该都是黑色
                # cv2.imshow(f'zero:{batch_idx}', img_zero_pad.numpy())
                # cv2.waitKey(0)
                return img_zero_pad, crop_pos

            # 根据mask,对周围进行零填充
            color_zero_pad, crop_pos = zero_padding_by_mask(color, mask_bs[batch_idx])
            color_bs_zero_pad[batch_idx] = color_zero_pad
            timer_1.tick('single batch | zero_padding')
            # 为下一帧计算mask_add
            mask_bs_next[batch_idx] = add_border_bool_by_crop_pos(mask, crop_pos, kernel_size=10)
            timer_1.tick('single batch | add border')
        timer.tick('go through batch and backproject pcd')
        # 提取几何特征
        points_feat_bs = points_feat_bs.type(torch.float32)  # points_feat_bs (bs, 1024, 3)
        points_feat = self.instance_geometry(points_feat_bs[:, :, :3])  # points_feat (bs, 64, 1024)
        timer.tick('get geometry feature')

        # 图像输入PSP-Net, 提取RGB特征
        color_bs_zero_pad = color_bs_zero_pad.cuda()
        color_bs_zero_pad = color_bs_zero_pad.type(torch.float32)
        # SGPA中,这里的图像已经被裁剪为192x192了
        # 之后可以测试一下
        # 这里绝对是可以继续优化的，因为后面choose会将没用的筛选掉，所以这里会计算很多没用CNN
        # (bs, 3, 640, 480) -> (bs, 32, 640, 480)
        out_img = self.psp(color_bs_zero_pad.permute(0, 3, 1, 2))
        di = out_img.size()[1]  # 特征维度 32
        emb = out_img.view(bs, di, -1)  # 将每张图像的特征变成每个像素点的
        timer.tick('get RGB feature CNN')

        # 对特征进行采样,采样n_pts个点
        # (bs, n_pts) -> (bs, n_dim=32, n_pts)
        choose_bs = choose_bs.transpose(1, 2).repeat(1, di, 1)
        # 根据choose提取对应点的特征
        emb = torch.gather(emb, 2, choose_bs).contiguous()
        # emb (bs, 64, n_pts)
        # emb (10, 64, 1024)
        # 1024个点的颜色特征，每个点的特征有64维
        emb = self.instance_color(emb)
        timer.tick('get RGB feature sampling')

        # 至此已经得到了(bs, 64, 1024)的几何特征与(bs, 64, 1024)的颜色特征
        # 将两者拼接得到instance local特征
        inst_local = torch.cat((points_feat, emb), dim=1)   # bs x 128 x 1024
        inst_global = self.instance_global(inst_local)      # bs x 1024 x 1
        timer.tick('get RGB feature concat')
        # 还需要做分割mask_for_next
        #
        # kps_3d = None
        # mask_bs_next 给下一帧使用的mask
        return inst_local, inst_global, mask_bs_next, points_bs

    def forward(self):
        # 传入的data分为两部分, 第一帧和后续帧
        # 1.初始帧
        # 是否添加噪声 if else
        init_frame = self.feed_dict[0]

        # 根据mask切分color
        init_pre_fetched = init_frame['meta']['pre_fetched']
        init_masks = init_pre_fetched['mask_add']
        init_colors = init_pre_fetched['color']
        init_nrms = init_pre_fetched['nrm']
        batch_size = len(init_masks)
        init_crop_pos = []
        # 可以对比的点:
        # last_frame 用mask
        # next_frame 用mask_add
        # for i in range(batch_size):
        #     idx = torch.where(init_masks[i, :, :])
        #     x_max = max(idx[1])
        #     x_min = min(idx[1])
        #     y_max = max(idx[0])
        #     y_min = min(idx[0])
        #     crop_pos_tmp = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
        #     init_crop_pos.append(crop_pos_tmp)
        use_ = False
        # 暂时不使用的代码
        if use_:
            pass
            # sift
            # for i in range(1, len(self.feed_dict)):
            #     last_frame = self.feed_dict[i - 1]
            #     next_frame = self.feed_dict[1]
            #     last_colors = last_frame['meta']['pre_fetched']['color']
            #     last_nrms = last_frame['meta']['pre_fetched']['nrm']
            #     next_colors = next_frame['meta']['pre_fetched']['color']
            #     next_nrms = next_frame['meta']['pre_fetched']['nrm']
            #     # sift匹配
            #     for j in range(batch_size):
            #         last_crop_color = crop_img(last_colors[j], crop_pos[j])
            #         next_crop_color = crop_img(next_colors[j], crop_pos[j])
            #         timer = Timer(True)
            #         color_sift_1, kp_xys_1, des_1 = extract_sift_kp_from_RGB(last_crop_color)
            #         color_sift_2, kp_xys_2, des_2 = extract_sift_kp_from_RGB(next_crop_color)
            #         timer.tick('sift feature extract')
            #         matches = sift_match(des_1, des_2, self.matcher)
            #         timer.tick('sift match ')
            #         # 可以用RANSAC过滤特征点
            #         # https://blog.csdn.net/sinat_41686583/article/details/115186277
            #
            #         # 取对应的normal map
            #         last_crop_nrm = crop_img(last_nrms[j], crop_pos[j])
            #         next_crop_nrm = crop_img(next_nrms[j], crop_pos[j])
            #         # 可视化
            #         # cv2.imshow('color_sift_1', color_sift_1)
            #         # cv2.waitKey(0)
            #         # cv2.imshow('color_sift_2', color_sift_2)
            #         # cv2.waitKey(0)
            #         #
            #         # cv2.imshow('nrm_1', norm2bgr(last_crop_nrm))
            #         # cv2.waitKey(0)
            #         # cv2.imshow('nrm_2', norm2bgr(next_crop_nrm))
            #         # cv2.waitKey(0)
            #
            #         # 提取3D点并进行匹配
            #         # 提取xyz+RGB+normal特征进行匹配


        # try FFB6D extract 3D kp
        mask_last_frame = self.feed_dict[0]['meta']['pre_fetched']['mask']
        crop_pos_last_frame = init_crop_pos   # 记录每张图像的四个裁剪坐标

        points_assign_mat = []
        # 第i帧
        for i in range(1, len(self.feed_dict)):
            last_frame = self.feed_dict[i - 1]
            next_frame = self.feed_dict[1]
            last_colors = last_frame['meta']['pre_fetched']['color']
            last_nrms = last_frame['meta']['pre_fetched']['nrm']
            next_colors = next_frame['meta']['pre_fetched']['color']
            next_nrms = next_frame['meta']['pre_fetched']['nrm']
            # 提取两帧的3D关键点
            # 提取第一帧的关键点
            print('extracting feature 1  ...')
            timer_extract_feat = Timer(True)
            inst_local_1, inst_global_1, mask_bs_next, points_bs_1 = self.extract_3D_kp(last_frame, mask_last_frame)
            timer_extract_feat.tick('extract feature 1 end')

            print('extracting feature 2  ...')
            inst_local_2, inst_global_2, _, points_bs_2 = self.extract_3D_kp(next_frame, mask_bs_next)
            timer_extract_feat.tick('extract feature 2 end')

            # 参考SGPA计算对应矩阵A
            # 1 -> 2
            # pts1 = A1*pts2
            assign_matrix_bs_1 = self.get_assgin_matrix(inst_local_1, inst_global_1, inst_global_2)
            timer_extract_feat.tick('get_assgin_matrix 1 end')
            # 2 -> 1
            # pts2 = A2*pts1
            assign_matrix_bs_2 = self.get_assgin_matrix(inst_local_2, inst_global_2, inst_global_1)
            timer_extract_feat.tick('get_assgin_matrix 2 end')

            points_assign_mat.append((points_bs_1, points_bs_2, assign_matrix_bs_1, assign_matrix_bs_2))
            # len(points_assign_mat)是帧的数量-1
        return points_assign_mat
        # if i != len(self.feed_dict):
        #     # 对于不同大小crop CNN的处理
        #     # 第一帧就使用addborder(但是对于add的部分是否要调整为0？)
        #     # 第二帧用第一帧的addborder
        #     # 计算出位姿后,重新计算mask
        #     # 还有后续帧
        #     # 将第一帧mask后的点云通过RT变换到第二帧,来为之后的帧提供mask
        #     mask_last_frame = get_mask()

        # bs = embedding.size()[0]
        # out = F.relu(self.fc1(embedding))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        # out_pc = out.view(bs, -1, 3)
        # return out_pc


    def set_data(self, data):
        print('set_data ...')
        # 提取需要的数据,并转移到cpu,并使用新的字典来存储
        self.feed_dict = []
        # self.npcs_feed_dict = []
        for i, frame in enumerate(data):
            if i == 0:
                self.feed_dict.append(self.convert_init_frame_data(frame))
            else:
                self.feed_dict.append(self.convert_subseq_frame_data(frame))
            # self.npcs_feed_dict.append(self.convert_subseq_frame_npcs_data(frame))
        print('set_data end')

    def update(self):
        print('forwarding ...')
        points_assign_mat = self.forward()
        print('forward end')
        # 计算loss
        if self.mode == 'train':
            print('computing loss')
            loss = self.criterion(points_assign_mat)
            print('compute loss end')
            print(f'loss: {loss}')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('backward end')
        # 调用forward,并计算loss
        # self.forward(save=save)
        # if not no_eval:
        #     self.compute_loss(test=True, per_instance=save, eval_iou=True, test_prefix='test')
        # else:
        #     self.loss_dict = {}

    def test(self, data):
        self.eval()
        self.set_data(data)
        points_assign_mat = self.forward()
        return points_assign_mat
