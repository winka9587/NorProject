import torch
import torch.nn as nn
import torch.nn.functional as F
from part_dof_utils import part_model_batch_to_part, eval_part_full, add_noise_to_part_dof, \
    compute_parts_delta_pose
from utils import cvt_torch, Timer, add_border_bool
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


class SIFT_Track(nn.Module):
    def __init__(self, device, real, subseq_len=2):
        super(SIFT_Track, self).__init__()
        # self.fc1 = nn.Linear(emb_dim, 512)
        # self.fc2 = nn.Linear(512, 1024)
        # self.fc3 = nn.Linear(1024, 3*n_pts)
        self.subseq_len = subseq_len
        self.device = device
        self.num_parts = 1
        self.num_joints = 0
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        if real:
            self.intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        else:
            self.intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

        # SGPA
        self.psp = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')

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


    # 有mask_add_from_last_frame说明是第二帧，需要使用前一帧提供的mask
    def extract_3D_kp(self, frame, mask_bs):
        # rgb                彩色图像               [bs, 3, h, w]
        # dpt_nrm            图像:xyz+normal       [bs, 6, h, w], 3c xyz in meter + 3c normal map
        # cld_rgb_nrm点云:    xyz+rgb+normal      [bs, 9, npts]
        # choose             应该是mask            [bs, 1, npts]

        # 输入color,normal,depth,mask,反投影，得到9通道的点云
        color_bs = frame['meta']['pre_fetched']['color']
        depth_bs = frame['meta']['pre_fetched']['depth']
        nrm_bs = frame['meta']['pre_fetched']['nrm']

        # 遍历batch
        bs = len(depth_bs)
        mask_bs_next = mask_bs.clone()
        for batch_idx in range(bs):
            color = color_bs[batch_idx]
            depth = depth_bs[batch_idx]
            nrm = nrm_bs[batch_idx]
            mask = mask_bs[batch_idx]
            # 1.提取几何特征
            # 根据mask,反投影得到观测点云
            points, idxs = backproject(depth, self.intrinsics, mask=mask_bs[batch_idx])
            points = torch.from_numpy(points)
            points_rgb = color[idxs[0], idxs[1]]
            points_nrm = nrm[idxs[0], idxs[1]]
            pts_9d = torch.concat([points, points_rgb, points_nrm], dim=1)


            # 2.提取颜色特征
            # 通过mask获得crop_pos
            def get_crop_pos_by_mask(mask):
                crop_idx = torch.where(mask)
                x_max = max(crop_idx[1]).item()
                x_min = min(crop_idx[1]).item()
                y_max = max(crop_idx[0]).item()
                y_min = min(crop_idx[0]).item()
                crop_pos_tmp = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
                return crop_pos_tmp
            def zero_padding_by_mask(img, mask):
                crop_pos = get_crop_pos_by_mask(mask)
                img_zero_pad = torch.zeros(img.shape, dtype=img.dtype)
                # 创建idx[0] y
                # 创建indx[1] x

                idx1 = torch.arange(crop_pos['x_min'], crop_pos['x_max'])
                idx0 = torch.full(idx1.shape, crop_pos['y_min'])
                mask_crop_idx0 = idx0
                mask_crop_idx1 = idx1
                for y in range(1, crop_pos['y_max']-crop_pos['y_min']):
                    idx0 = torch.full(idx1.shape, crop_pos['y_min']+y)
                    mask_crop_idx0 = torch.concat((mask_crop_idx0, idx0))
                    mask_crop_idx1 = torch.concat((mask_crop_idx1, idx1))
                mask_crop_idx = (mask_crop_idx0, mask_crop_idx1)
                img_zero_pad[mask_crop_idx] = img[mask_crop_idx]
                cv2.imshow('zero', img_zero_pad.numpy())
                cv2.waitKey(0)
                return img_zero_pad

            # 根据mask,对周围进行零填充
            color_zero_pad = zero_padding_by_mask(color, mask_bs[batch_idx])
            # 为下一帧计算mask_add
            mask_bs_next[batch_idx] = add_border_bool(mask, kernel_size=10)

            # 图像输入PSP-Net
            out_img = self.psp(color_zero_pad)
            di = out_img.size()[1]
            emb = out_img.view(bs, di, -1)
            choose = choose.unsqueeze(1).repeat(1, di, 1)
            emb = torch.gather(emb, 2, choose).contiguous()
            emb = self.instance_color(emb)




        # 还需要做分割mask_for_next
        #
        kps_3d = None
        return kps_3d, mask_bs_next

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
            # sift
            for i in range(1, len(self.feed_dict)):
                last_frame = self.feed_dict[i - 1]
                next_frame = self.feed_dict[1]
                last_colors = last_frame['meta']['pre_fetched']['color']
                last_nrms = last_frame['meta']['pre_fetched']['nrm']
                next_colors = next_frame['meta']['pre_fetched']['color']
                next_nrms = next_frame['meta']['pre_fetched']['nrm']
                # sift匹配
                for j in range(batch_size):
                    last_crop_color = crop_img(last_colors[j], crop_pos[j])
                    next_crop_color = crop_img(next_colors[j], crop_pos[j])
                    timer = Timer(True)
                    color_sift_1, kp_xys_1, des_1 = extract_sift_kp_from_RGB(last_crop_color)
                    color_sift_2, kp_xys_2, des_2 = extract_sift_kp_from_RGB(next_crop_color)
                    timer.tick('sift feature extract')
                    matches = sift_match(des_1, des_2, self.matcher)
                    timer.tick('sift match ')
                    # 可以用RANSAC过滤特征点
                    # https://blog.csdn.net/sinat_41686583/article/details/115186277

                    # 取对应的normal map
                    last_crop_nrm = crop_img(last_nrms[j], crop_pos[j])
                    next_crop_nrm = crop_img(next_nrms[j], crop_pos[j])
                    # 可视化
                    # cv2.imshow('color_sift_1', color_sift_1)
                    # cv2.waitKey(0)
                    # cv2.imshow('color_sift_2', color_sift_2)
                    # cv2.waitKey(0)
                    #
                    # cv2.imshow('nrm_1', norm2bgr(last_crop_nrm))
                    # cv2.waitKey(0)
                    # cv2.imshow('nrm_2', norm2bgr(next_crop_nrm))
                    # cv2.waitKey(0)

                    # 提取3D点并进行匹配
                    # 提取xyz+RGB+normal特征进行匹配


        # try FFB6D extract 3D kp
        mask_last_frame = self.feed_dict[0]['meta']['pre_fetched']['mask']
        crop_pos_last_frame = init_crop_pos   # 记录每张图像的四个裁剪坐标
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
            mask_add = self.extract_3D_kp(last_frame, mask_last_frame)
            self.extract_3D_kp(next_frame, mask_add)
            if i != len(self.feed_dict):
                # 对于不同大小crop CNN的处理
                # 第一帧就使用addborder(但是对于add的部分是否要调整为0？)
                # 第二帧用第一帧的addborder
                # 计算出位姿后,重新计算mask
                # 还有后续帧
                # 将第一帧mask后的点云通过RT变换到第二帧,来为之后的帧提供mask
                mask_last_frame = get_mask()










        # bs = embedding.size()[0]
        # out = F.relu(self.fc1(embedding))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        # out_pc = out.view(bs, -1, 3)
        # return out_pc


    def set_data(self, data):
        # 提取需要的数据,并转移到cpu,并使用新的字典来存储
        self.feed_dict = []
        self.npcs_feed_dict = []
        for i, frame in enumerate(data):
            if i == 0:
                self.feed_dict.append(self.convert_init_frame_data(frame))
            else:
                self.feed_dict.append(self.convert_subseq_frame_data(frame))
            self.npcs_feed_dict.append(self.convert_subseq_frame_npcs_data(frame))

    def update(self):
        self.forward()
        # 调用forward,并计算loss
        # self.forward(save=save)
        # if not no_eval:
        #     self.compute_loss(test=True, per_instance=save, eval_iou=True, test_prefix='test')
        # else:
        #     self.loss_dict = {}
        pass


