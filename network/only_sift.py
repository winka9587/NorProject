import torch
import torch.nn as nn
import torch.nn.functional as F
from part_dof_utils import part_model_batch_to_part, eval_part_full, add_noise_to_part_dof, \
    compute_parts_delta_pose
from utils import cvt_torch2, Timer, add_border_bool_by_crop_pos, get_bbox
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
from network.lib.utils import sample_points_from_mesh, render_points_diff_color

from lib.pspnet import PSPNet
from lib.pointnet import Pointnet2MSG

from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from lib.utils import render_points_diff_color

from visualize import viz_multi_points_diff_color, viz_mask_bool
import normalSpeed

norm_color = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


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

# 通过mask获得crop_pos
def get_crop_pos_by_mask(mask):
    crop_idx = torch.where(mask)
    cmax = torch.max(crop_idx[1]).item()
    cmin = torch.min(crop_idx[1]).item()
    rmax = torch.max(crop_idx[0]).item()
    rmin = torch.min(crop_idx[0]).item()
    rmin, rmax, cmin, cmax = get_bbox((rmin, cmin, rmax, cmax))
    crop_pos_tmp = {'cmin': cmin, 'cmax': cmax, 'rmin': rmin, 'rmax': rmax}
    return crop_pos_tmp


# 通过mask划分bbox，然后对bbox外的区域进行零填充
def zero_padding_by_mask(img, mask):
    t2 = Timer(True)
    bbox = get_crop_pos_by_mask(mask)
    t2.tick('--get crop by mask')
    img_zero_pad = torch.zeros(img.shape, dtype=img.dtype)
    # 创建idx[0] y
    # 创建indx[1] x

    idx1 = torch.arange(bbox['cmin'], bbox['cmax'])
    idx0 = torch.full(idx1.shape, bbox['rmin'])
    t2.tick('--fill zero')
    mask_crop_idx0 = idx0
    mask_crop_idx1 = idx1
    for y in range(1, bbox['rmax'] - bbox['rmin']):
        idx0 = torch.full(idx1.shape, bbox['rmin'] + y)
        mask_crop_idx0 = torch.concat((mask_crop_idx0, idx0))
        mask_crop_idx1 = torch.concat((mask_crop_idx1, idx1))
    mask_crop_idx = (mask_crop_idx0, mask_crop_idx1)
    img_zero_pad[mask_crop_idx] = img[mask_crop_idx]
    t2.tick('--map mask value to img')
    # 可视化零填充后的mask,bbox之外应该都是黑色
    # cv2.imshow(f'zero:{batch_idx}', img_zero_pad.numpy())
    # cv2.waitKey(0)
    return img_zero_pad, bbox


# 从mask中提取n_pts个点的下标 choose
def choose_from_mask_bs(mask_bs, n_pts):
    output_choose = torch.tensor([])
    mask_bs_choose = torch.full(mask_bs.shape, False)
    for b_i in range(len(mask_bs)):
        # get choose pixel index by mask
        mask = mask_bs[b_i]
        mask_flatten = torch.full(mask.flatten().shape, False)  # 为了确定采样点的像素坐标
        choose = torch.where(mask.flatten())[0]
        if len(choose) > n_pts:
            choose_mask = np.zeros(len(choose), dtype=np.uint8)
            choose_mask[:n_pts] = 1
            np.random.shuffle(choose_mask)
            choose_mask = torch.from_numpy(choose_mask)
            choose = choose[choose_mask.nonzero()]
        else:
            # 点的数量不够,需要补齐n_pts个点
            # wrap 用前面补后面，用后面补前面
            choose = torch.from_numpy(np.pad(choose.numpy(), (0, n_pts - len(choose)), 'wrap'))
        mask_flatten[choose] = True
        mask_flatten = mask_flatten.view(mask.shape).numpy()
        # 可视化mask
        # viz_mask_bool('mask_flatten', mask_flatten)
        choose = torch.unsqueeze(choose, 0)  # (n_pts, ) -> (1, n_pts)
        output_choose = torch.cat((output_choose, choose), 0)
        output_choose = output_choose.type(torch.int64)
    return output_choose


class SIFT_Track(nn.Module):
    def __init__(self, real, subseq_len=2, mode='train', opt=None, img_size=192, remove_border_w=5, tb_writer=None):
        self.writer = tb_writer  # tensorboard writer
        super(SIFT_Track, self).__init__()
        # self.fc1 = nn.Linear(emb_dim, 512)
        # self.fc2 = nn.Linear(512, 1024)
        # self.fc3 = nn.Linear(1024, 3*n_pts)
        self.remove_border_w = remove_border_w
        self.mode = mode
        self.subseq_len = subseq_len
        # self.device = device
        self.num_parts = 1
        self.num_joints = 0
        # self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.img_process = "crop" # zero_padding(对不关心区域进行零填充), crop(裁剪并resize)
        self.img_size = img_size

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
        self.instance_rgbFeature = Pointnet2MSG(0)
        self.instance_nrmFeature = Pointnet2MSG(0)
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
                item = item.long().cuda()
                # item = item.long().to(self.device)
            else:
                item = item.float().cuda()
                # item = item.float().to(self.device)
            feed_frame[key] = item
        return feed_frame


    # 最终返回的input包含:(带*是相比于init frame多出来的)
    # points    (B,3,4096)
    # meta      (B,3,1)
    # labels    (B,4096)
    # gt_part   (rotation,scale,translation)
    # points_mean *     (10,3,1)
    # npcs *            (10,3,4096)
    def convert_subseq_frame_data(self, data):
        # 观测点云,观测点云平均点,gt位姿
        input = {'points': data['points'],
                 'points_mean': data['meta']['points_mean']}
        # 添加nocs点云
        if 'nocs' in data:
            input['nocs'] = data['nocs']
        input = cvt_torch2(input)
        # 添加meta
        input['meta'] = data['meta']
        # 添加labels
        if 'labels' in data:
            input['labels'] = data['labels'].long().cuda()
        return input

    # 最终的input包含:
    # points
    # nocs
    # labels
    # meta
    # points_mean
    # def convert_subseq_frame_npcs_data(self, data):
    #     input = {}
    #     for key, item in data.items():
    #         if key not in ['meta', 'labels', 'points', 'nocs']:
    #             continue
    #         elif key in ['meta']:
    #             pass
    #         elif key in ['labels']:
    #             item = item.long().cuda()
    #         else:
    #             item = item.float().cuda()
    #         input[key] = item
    #     input['points_mean'] = data['meta']['points_mean'].float().cuda()
    #     return input

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

    def remove_border_bool(self, mask, kernel_size=5):  # enlarge the region w/ 255
        # print((255 - mask).sum())
        if isinstance(mask, torch.Tensor):
            output = mask.clone()
        else:
            output = mask.copy()
        h, w = mask.shape
        for i in range(h):
            for j in range(w):
                if mask[i][j] == False:
                    output[max(0, i - kernel_size): min(h, i + kernel_size),
                    max(0, j - kernel_size): min(w, j + kernel_size)] = False
        # print((255 - output).sum())
        return output

    # 有mask_add_from_last_frame说明是第二帧，需要使用前一帧提供的mask
    def extract_3D_kp(self, frame, mask_bs):
        timer = Timer(True)
        # rgb                彩色图像               [bs, 3, h, w]
        # choose             应该是mask            [bs, 1, npts]

        # 输入color,normal,depth,mask,反投影，得到9通道的点云
        color_bs = frame['meta']['pre_fetched']['color']
        depth_bs = frame['meta']['pre_fetched']['depth']

        timer.tick('extract idx from mask')

        # 遍历batch
        bs = len(depth_bs)
        mask_bs_next = mask_bs.clone()
        # img_bs = color_bs.clone()
        img_bs = []
        choose_bs = []
        points_feat_bs = torch.tensor([]).cuda()  # 提取的几何特征
        points_bs = torch.tensor([]).cuda()  # (bs, 1024, 3)
        points_origin_bs = torch.tensor([]).cuda()
        timer_1 = Timer(True)
        for batch_idx in range(bs):
            color = color_bs[batch_idx]
            depth = depth_bs[batch_idx]
            mask = mask_bs[batch_idx]

            if self.remove_border_w > 0:
                mask = self.remove_border_bool(mask, kernel_size=self.remove_border_w)


            # 先通过mask得到bbox，然后对所有图像进行裁剪
            if self.img_process == "zero_padding":
                # 根据mask,对周围进行零填充
                color_zero_pad, bbox = zero_padding_by_mask(color, mask_bs[batch_idx])
                # img_bs[batch_idx] = color_zero_pad
                img_bs.append(color_zero_pad)
                timer_1.tick('single batch | zero_padding')
            if self.img_process == "crop":
                # 根据mask,对RGB进行裁剪
                bbox = get_crop_pos_by_mask(mask_bs[batch_idx])
                rmin = bbox['rmin']
                rmax = bbox['rmax']
                cmin = bbox['cmin']
                cmax = bbox['cmax']
                img_cropped = color[rmin:rmax, cmin:cmax].cpu().numpy()
                img_cropped = cv2.resize(img_cropped, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                img_cropped = norm_color(img_cropped)
                # img_bs[batch_idx] = img_cropped
                img_bs.append(img_cropped)
                timer_1.tick('single batch | crop img')

            # 从mask获得choose
            mask = torch.logical_and(mask, depth > 0)
            # choose是bbox裁剪后的mask生成的，但是可以直接用来读取裁剪后图像上的坐标
            # 如果想要读取原大小图像上的像素点， 会有问题
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()
            if len(choose) < 32:
                print('available pixel less than 32, cant use this')
                continue
            if len(choose) > self.n_pts:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.n_pts] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.n_pts - len(choose)), 'wrap')


            # 1.提取几何特征

            # 计算normal map在 划分bbox之后
            if 'real' in self.mode:
                fx = 591.0125
                fy = 590.16775
            else:
                fx = 577.5
                fy = 577.5
            k_size = 3
            distance_threshold = 2000
            difference_threshold = 20  # 周围的点深度距离超过10mm则不考虑
            point_into_surface = False
            # 使用裁剪后的深度图计算normal map
            depth_nor = depth[rmin:rmax, cmin:cmax].numpy().astype(np.uint16)
            normals_map_out = normalSpeed.depth_normal(depth_nor, fx, fy, k_size, distance_threshold, difference_threshold,
                                                       point_into_surface)
            timer.tick('depth 2 normal_end')
            nrm = normals_map_out

            # width_crop = depth[rmin:rmax, cmin:cmax].shape[1]
            # height_crop = depth[rmin:rmax, cmin:cmax].shape[0]


            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]       # 对点云进行一个采样,采样n_pts个点
            # 用来提取裁剪前图像上的点，主要用于反投影(因为choose是在裁剪前的mask的2D bbox上得到的)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]     # 像素坐标u
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]     # 像素坐标v

            # origin backproject code of NorProject, may from SGPA.
            # pt2 = depth_masked / self.norm_scale
            # pt2 = pt2.numpy()
            # pt0 = (xmap_masked - self.cam_cx) * pt2 / self.cam_fx     # x
            # pt1 = (ymap_masked - self.cam_cy) * pt2 / self.cam_fy     # y
            # points = np.concatenate((pt0, pt1, pt2), axis=1)  # xyz

            # new backproject code
            # like code of CAPTRA, use -z
            pt2 = depth_masked
            pt2 = pt2.numpy()
            pt0 = (xmap_masked - self.cam_cx) * pt2 / self.cam_fx  # x
            # uv: v = height-v
            ymap_masked2 = self.ymap.shape[0] - ymap_masked
            pt1 = (ymap_masked2 - self.cam_cy) * pt2 / self.cam_fy  # y
            pt2 = -pt2
            points = np.concatenate((pt0, pt1, pt2), axis=1)
            points = points / self.norm_scale

            # 用来提取裁剪后图像上的点，用于从rgb和nrm上提取点云对应的特征
            xmap_crop_masked = xmap_masked - cmin - 1
            ymap_crop_masked = ymap_masked - rmin - 1

            crop_w = rmax - rmin
            ratio = self.img_size / crop_w
            col_idx = choose % crop_w  # 一维坐标转裁剪后的二维坐标
            row_idx = choose // crop_w
            # 计算裁剪后的二维坐标到resize后的二维坐标，然后转一维坐标
            choose_resize = (torch.floor(row_idx * ratio) * self.img_size + torch.floor(col_idx * ratio)).type(torch.int64)
            choose_bs.append(choose_resize)  # 存放resize图像对应的choose (1024, 1)

            # points_viz = points.copy()  # 测试反投影|(1)
            points = torch.from_numpy(np.transpose(points, (2, 0, 1))).cuda()  # points (1024, 3, 1) -> (1, 1024, 3)
            points = points.type(torch.float32)
            points_origin_bs = torch.cat((points_origin_bs, points), 0)  # 未均值化的点云
            # 使用均值进行归一化
            # points_mean (10, 3, 1)
            # points (1, 1024, 3)
            points_mean_single_batch = torch.mean(points, axis=1, keepdims=True).cuda().repeat(1, points.shape[1], 1)
            # 这个points_mean是整个batch的points的mean
            # points_mean_single_batch = frame['meta']['points_mean'][batch_idx].cuda().unsqueeze(0)
            points = points - points_mean_single_batch

            points_bs = torch.cat((points_bs, points), 0)  # 均值化的点云
            timer_1.tick('single batch | backproject')

            # if self.mode == 'test':
            #     # 测试反投影|(2)可视化color 和 mask
            #     print('data from {}'.format(frame['meta']['ori_path']))
            #     cv2.imshow('rgb', color.clone().cpu().numpy())
            #     cv2.waitKey(0)
            #     viz_mask_bool('viz_mask', mask)
            #     # 测试反投影|(3)可视化裁剪的rgb
            #     rgb_show = torch.zeros(color.shape, dtype=torch.uint8)
            #     rgb_show[ymap_masked, xmap_masked, 0] = color[ymap_masked, xmap_masked, 0]
            #     rgb_show[ymap_masked, xmap_masked, 1] = color[ymap_masked, xmap_masked, 1]
            #     rgb_show[ymap_masked, xmap_masked, 2] = color[ymap_masked, xmap_masked, 2]
            #     cv2.imshow('rgb', rgb_show.numpy())
            #     cv2.waitKey(0)
            #     # 测试反投影|(3)可视化反投影点云
            #     points_viz = np.squeeze(np.transpose(points_viz, (2, 0, 1)), 0)
            #     color_red = np.array([255, 0, 0])
            #     color_green = np.array([0, 255, 0])
            #     color_blue = np.array([0, 0, 255])
            #     pts_colors = [color_red]
            #     render_points_diff_color('test backproject', [points_viz],
            #                              pts_colors, save_img=False,
            #                              show_img=True)

            # img_cropped (3, 192, 192)
            # 有一个问题，img_cropped是通过resize才变成192x192的，xmap_masked和ymap_masked应该如何变换？
            # 可以用未裁剪的图像+bbox+xmap来获得
            # 真正的问题在于emb，但是有一个ratio能否解决这个问题？

            points_rgb = color[ymap_masked, xmap_masked, :]
            points_rgb = points_rgb.squeeze(2).transpose(0, 1).cuda()  # points_rgb -> (1, 1024, 3)
            points_nrm = nrm[ymap_crop_masked, xmap_crop_masked]
            points_nrm = torch.from_numpy(points_nrm).squeeze(1).transpose(1, 0).cuda()  # points_nrm (1024, 1, 1, 3) -> (1, 1024, 3)
            timer_1.tick('single batch | extract color and nrm feature')

            # 如果要增加对噪声点的过滤，需要重新采样(重新计算choose)，或者在计算choose之前就进行过滤
            # viz_multi_points_diff_color(f'pts:{batch_idx}', [points], [np.array([[1, 0, 0]])])
            # 拼接三个不同的特征
            # 是否可以留着nrm在损失函数里使用？
            #  -> (1, 1024, 3) ->  -> (1, 1024, 9)
            pts_9d = torch.concat([points, points_rgb, points_nrm], dim=2)
            # 在这之前points_feat_bs 是空的
            points_feat_bs = torch.cat((points_feat_bs, pts_9d), 0)  #######对比SGPA，这里绝对不行！！！ 还有后面instance_geometry只输入了3维
            timer_1.tick('single batch | concat 3 feat')


            # 为下一帧计算mask_add
            mask_bs_next[batch_idx] = add_border_bool_by_crop_pos(mask, bbox, kernel_size=10)
            timer_1.tick('single batch | add border')
            # 这里只有加入分割对应网络

        timer.tick('go through batch and backproject pcd')
        # 提取几何特征
        points_feat_bs = points_feat_bs.type(torch.float32)  # points_feat_bs (bs, 1024, 3)
        points_feat = self.instance_geometry(points_feat_bs[:, :, :3])  # points_feat (bs, 64, 1024)

        timer.tick('get geometry feature')
        points_rgb_feat = self.instance_rgbFeature(points_feat_bs[:, :, 3:6])  # (bs, 64, 1024)
        points_nrm_feat = self.instance_nrmFeature(points_feat_bs[:, :, 6:])  # (bs, 64, 1024)

        # 图像输入PSP-Net, 提取RGB特征
        img_bs = torch.stack(img_bs, dim=0).cuda()  # convert list to tensor
        img_bs = img_bs.type(torch.float32)
        # SGPA中,这里的图像已经被裁剪为192x192了
        # 之后可以测试一下
        # problem: 这里绝对是可以继续优化的，因为后面choose会将没用的筛选掉，所以这里会计算很多没用CNN

        # (bs, 3, 192, 192) -> (bs, 32, 192, 192)
        out_img = self.psp(img_bs)
        di = out_img.size()[1]  # 特征维度 32
        emb = out_img.view(bs, di, -1)  # 将一张图像的特征变成图像上每个像素点的
        timer.tick('get RGB feature CNN')

        # choose_bs list 转 tensor
        choose_bs = torch.stack(choose_bs, dim=0).cuda()  # convert list to tensor
        choose_bs = choose_bs.type(torch.int64)

        # 弃用
        # 对特征进行采样,采样n_pts个点
        # choose_bs (bs, n_pts=1024, 1) -> (bs, n_dim=32, n_pts=1024)
        # choose_bs = choose_bs.transpose(1, 2).repeat(1, di, 1)
        # 根据choose提取对应点的特征
        choose_bs = choose_bs.permute(0, 2, 1).repeat(1, di, 1)  # (bs=10, n_pts=1024, 1) -> (bs=10, di=32, n_pts=1024)
        emb = torch.gather(emb, 2, choose_bs).contiguous()
        # emb (bs, 64, n_pts)
        # emb (10, 64, 1024)
        # 1024个点的颜色特征，每个点的特征有64维
        emb = self.instance_color(emb)
        timer.tick('get RGB feature sampling')

        # 至此已经得到了(bs, 64, 1024)的几何特征与(bs, 64, 1024)的颜色特征
        # 将两者拼接得到instance local特征
        # 检查一下几何特征与颜色特征的值域
        # 颜色特征使用卷积结果
        inst_local = torch.cat((points_feat, emb), dim=1)   # bs x 128 x 1024

        # 卷积结果替换为pointnet的结果
        # inst_local = torch.cat((points_feat, points_rgb_feat), dim=1)   # bs x (64+64) x 1024 -> bs x 128 x 1024

        inst_global = self.instance_global(inst_local)      # bs x 1024 x 1
        timer.tick('get RGB feature concat')
        # 还需要做分割mask_for_next
        #
        # kps_3d = None
        # mask_bs_next 给下一帧使用的mask
        return inst_local, inst_global, mask_bs_next, points_bs, points_origin_bs

    def forward(self, data):
        # 提取需要的数据,并转移到cpu,并使用新的字典来存储
        self.feed_dict = []
        # self.npcs_feed_dict = []
        for i, frame in enumerate(data):
            if i == 0:
                self.feed_dict.append(self.convert_init_frame_data(frame))
            else:
                self.feed_dict.append(self.convert_subseq_frame_data(frame))
        # 传入的data分为两部分, 第一帧和后续帧
        # 1.初始帧
        # 是否添加噪声 if else
        init_frame = self.feed_dict[0]

        # 根据mask切分color
        init_pre_fetched = init_frame['meta']['pre_fetched']
        init_masks = init_pre_fetched['mask_add']
        init_colors = init_pre_fetched['color']
        batch_size = len(init_masks)
        init_crop_pos = []

        # try FFB6D extract 3D kp
        mask_last_frame = self.feed_dict[0]['meta']['pre_fetched']['mask']
        crop_pos_last_frame = init_crop_pos   # 记录每张图像的四个裁剪坐标

        points_assign_mat = []
        gt_pose_between_frame = []
        # 第i帧
        for i in range(1, len(self.feed_dict)):
            last_frame = self.feed_dict[i - 1]
            next_frame = self.feed_dict[1]
            last_colors = last_frame['meta']['pre_fetched']['color']
            next_colors = next_frame['meta']['pre_fetched']['color']
            # 提取两帧的3D关键点
            # 提取第一帧的关键点
            print('extracting feature 1  ...')
            timer_extract_feat = Timer(True)
            inst_local_1, inst_global_1, mask_bs_next, points_bs_1, points_origin_bs_1 = self.extract_3D_kp(last_frame, mask_last_frame)
            timer_extract_feat.tick('extract feature 1 end')

            print('extracting feature 2  ...')


            # 测试用，使用第二帧的gt_mask，如果要测试前一阵的的padding mask，删除这一行
            mask_bs_next = self.feed_dict[i]['meta']['pre_fetched']['mask']



            inst_local_2, inst_global_2, _, points_bs_2, points_origin_bs_2 = self.extract_3D_kp(next_frame, mask_bs_next)
            timer_extract_feat.tick('extract feature 2 end')

            # 参考SGPA计算对应矩阵A
            # 1 -> 2
            # pts1 = A1*pts2
            assign_matrix_bs_1 = self.get_assgin_matrix(inst_local_1, inst_global_1, inst_global_2)  # 参考SGPA loss.py line 40: assign_matrix_bs_1 应该与points_1相乘
            timer_extract_feat.tick('get_assgin_matrix 1 end')
            # 2 -> 1
            # pts2 = A2*pts1
            assign_matrix_bs_2 = self.get_assgin_matrix(inst_local_2, inst_global_2, inst_global_1) # assign_matrix_bs_1 应该与points_1相乘
            timer_extract_feat.tick('get_assgin_matrix 2 end')

            points_assign_mat.append((points_bs_1, points_bs_2, assign_matrix_bs_1, assign_matrix_bs_2, points_origin_bs_1, points_origin_bs_2))
            # len(points_assign_mat)是帧的数量-1

            # 计算两帧之间的gt位姿
            pose_1_tmp = last_frame['meta']['nocs2camera'][0]
            pose_2_tmp = next_frame['meta']['nocs2camera'][0]

            # pose_1_bs
            #   rotation (10, 1, 3, 3)
            #   scale   (10, 1)
            #   translation     (10, 1, 3, 1)

            # pose_1_tmp
            #   rotation (10, 3, 3)
            #   scale   (10)
            #   translation     (10, 3, 1)
            # tmp
            pose_12_tmp = {}
            # scale    (10)
            pose_12_tmp['scale'] = pose_2_tmp['scale']/pose_1_tmp['scale']
            # rotation (10, 3, 3)
            pose_12_tmp['rotation'] = torch.bmm(pose_2_tmp['rotation'],
                                               pose_1_tmp['rotation'].transpose(-2, -1))
            # translation (10, 3, 1)
            pose_12_tmp['translation'] = pose_2_tmp['translation'] - \
                                        (pose_12_tmp['scale'].reshape(-1, 1, 1) * \
                                         torch.bmm(pose_12_tmp['rotation'],
                                                   pose_1_tmp['translation']))

            # tmp
            sRt_2_tmp = torch.eye(4)
            sRt_2_tmp[:3, :3] = pose_2_tmp['scale'][0] * pose_2_tmp['rotation'][0]
            sRt_2_tmp[:3, 3] = pose_2_tmp['translation'].squeeze(-1)[0]

            sRt_1_tmp = torch.eye(4)
            sRt_1_tmp[:3, :3] = pose_1_tmp['scale'][0] * pose_1_tmp['rotation'][0]
            sRt_1_tmp[:3, 3] = pose_1_tmp['translation'].squeeze(-1)[0]

            sRt_12_tmp = torch.eye(4).type(torch.float64)
            sRt_12_tmp[:3, :3] = pose_12_tmp['scale'][0] * pose_12_tmp['rotation'][0]
            sRt_12_tmp[:3, 3] = pose_12_tmp['translation'].squeeze(1).squeeze(-1)[0]

            debug = False
            # load model
            if debug:
                # 位姿并不是从模型坐标系到观测点云的，而是从nocs到观测点云的
                # 因此应该得到nocs才行
                # gt_model_numpy (2048, 3)
                gt_model_numpy = sample_points_from_mesh('/data1/cxx/Lab_work/dataset/obj_models/real_train/bottle_blue_google_norm.obj', 2048, fps=True, ratio=3)

                model = np.ones((2048, 4))
                model[:, :3] = gt_model_numpy

                m1 = (sRt_1_tmp.numpy() @ model.transpose()).transpose()[:, :3]
                m2 = (sRt_2_tmp.numpy() @ model.transpose()).transpose()[:, :3]

                model_1 = np.ones((2048, 4))
                model_1[:, :3] = m1
                m1_12 = (sRt_12_tmp.numpy() @ model_1.transpose()).transpose()[:, :3]

                render_points_diff_color("model", [gt_model_numpy], [np.array([255, 0, 0])])
                render_points_diff_color("p1 and p2", [m1, m2], [np.array([255, 0, 0]), np.array([0, 255, 0])])

                # tmp
                coord = last_frame['nocs'][0].transpose(-1, -2).cpu().numpy()
                coord_ = np.ones((4096, 4))
                coord_[:, :3] = coord

                coord_1 = (sRt_1_tmp.numpy() @ coord_.transpose()).transpose()
                coord_12 = (sRt_12_tmp.numpy() @ coord_1.transpose()).transpose()[:, :3]

                coord_1 = coord_1[:, :3]
                coord_2 = (sRt_2_tmp.numpy() @ coord_.transpose()).transpose()[:, :3]

                render_points_diff_color("coord 1, 2, 12", [coord_1, coord_2, coord_12], [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])])

        # 检查模型投影到图像上
        print(last_frame['meta']['path'])

        # 测试位姿相关
        if debug:
            debug_info = (m1, m2, gt_model_numpy, m1_12, coord_1, coord_2, coord_12)
        else:
            debug_info = None

        return points_assign_mat, pose_12_tmp, debug_info

    def test(self, data):
        self.eval()
        t = Timer(True)
        self.set_data(data)
        t.tick('[test]: set data end')
        points_assign_mat, pose12,_ = self.forward()
        t.tick('[test]: forward  end')
        return points_assign_mat, pose12
