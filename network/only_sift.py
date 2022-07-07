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

# ͨ��mask���crop_pos
def get_crop_pos_by_mask(mask):
    crop_idx = torch.where(mask)
    cmax = torch.max(crop_idx[1]).item()
    cmin = torch.min(crop_idx[1]).item()
    rmax = torch.max(crop_idx[0]).item()
    rmin = torch.min(crop_idx[0]).item()
    rmin, rmax, cmin, cmax = get_bbox((rmin, cmin, rmax, cmax))
    crop_pos_tmp = {'cmin': cmin, 'cmax': cmax, 'rmin': rmin, 'rmax': rmax}
    return crop_pos_tmp


# ͨ��mask����bbox��Ȼ���bbox���������������
def zero_padding_by_mask(img, mask):
    t2 = Timer(True)
    bbox = get_crop_pos_by_mask(mask)
    t2.tick('--get crop by mask')
    img_zero_pad = torch.zeros(img.shape, dtype=img.dtype)
    # ����idx[0] y
    # ����indx[1] x

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
    # ���ӻ��������mask,bbox֮��Ӧ�ö��Ǻ�ɫ
    # cv2.imshow(f'zero:{batch_idx}', img_zero_pad.numpy())
    # cv2.waitKey(0)
    return img_zero_pad, bbox


# ��mask����ȡn_pts������±� choose
def choose_from_mask_bs(mask_bs, n_pts):
    output_choose = torch.tensor([])
    mask_bs_choose = torch.full(mask_bs.shape, False)
    for b_i in range(len(mask_bs)):
        # get choose pixel index by mask
        mask = mask_bs[b_i]
        mask_flatten = torch.full(mask.flatten().shape, False)  # Ϊ��ȷ�����������������
        choose = torch.where(mask.flatten())[0]
        if len(choose) > n_pts:
            choose_mask = np.zeros(len(choose), dtype=np.uint8)
            choose_mask[:n_pts] = 1
            np.random.shuffle(choose_mask)
            choose_mask = torch.from_numpy(choose_mask)
            choose = choose[choose_mask.nonzero()]
        else:
            # �����������,��Ҫ����n_pts����
            # wrap ��ǰ�油���棬�ú��油ǰ��
            choose = torch.from_numpy(np.pad(choose.numpy(), (0, n_pts - len(choose)), 'wrap'))
        mask_flatten[choose] = True
        mask_flatten = mask_flatten.view(mask.shape).numpy()
        # ���ӻ�mask
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
        self.img_process = "crop" # zero_padding(�Բ�����������������), crop(�ü���resize)
        self.img_size = img_size

        if real:
            self.intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
            self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = 591.0125, 590.16775, 322.525, 244.11084
        else:
            self.intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])
            self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = 577.5, 577.5, 319.5, 239.5

        # SGPA
        self.psp = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')
        self.n_pts = 1024  # ��mask�в���1024���㣬��ȡ��ɫ����
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

        # ��ͶӰ��
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
    # ���շ��ص�feed_frame(Ҳ���ǳ�ʼ֡)����
    # points, labels, meta, nocs, gt_part(nocs2camera��gtλ��)
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


    # ���շ��ص�input����:(��*�������init frame�������)
    # points    (B,3,4096)
    # meta      (B,3,1)
    # labels    (B,4096)
    # gt_part   (rotation,scale,translation)
    # points_mean *     (10,3,1)
    # npcs *            (10,3,4096)
    def convert_subseq_frame_data(self, data):
        # �۲����,�۲����ƽ����,gtλ��
        input = {'points': data['points'],
                 'points_mean': data['meta']['points_mean']}
        # ���nocs����
        if 'nocs' in data:
            input['nocs'] = data['nocs']
        input = cvt_torch2(input)
        # ���meta
        input['meta'] = data['meta']
        # ���labels
        if 'labels' in data:
            input['labels'] = data['labels'].long().cuda()
        return input

    # ���յ�input����:
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

    # ���ƶ�Ӧ����
    # bs x n_pts x nv
    # (10, 1024, 1024)
    def get_assgin_matrix(self, inst_local, inst_global, cat_global):
        assign_feat_bs = torch.cat((inst_local, inst_global.repeat(1, 1, self.n_pts), cat_global.repeat(1, 1, self.n_pts)),
                                dim=1)  # bs x 2176 x n_pts
        assign_mat_bs = self.assignment(assign_feat_bs)  # bs x 1024 x 1024  (bs, n_pts_2, npts_1)
        # nvԭ����prior�������,��������Ϊ�õ���ʵ������(ֻ�ǲ�ͬ֡����),������self.n_pts��ͬ
        nv = self.n_pts
        assign_mat_bs = assign_mat_bs.view(-1, nv, self.n_pts).contiguous()  # bs, nv, n_pts -> bs, nv, n_pts

        # assign_mat = torch.index_select(assign_mat, 0, index)  # ɾ��softmax��һ����
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

    # ��mask_add_from_last_frame˵���ǵڶ�֡����Ҫʹ��ǰһ֡�ṩ��mask
    def extract_3D_kp(self, frame, mask_bs):
        timer = Timer(True)
        # rgb                ��ɫͼ��               [bs, 3, h, w]
        # choose             Ӧ����mask            [bs, 1, npts]

        # ����color,normal,depth,mask,��ͶӰ���õ�9ͨ���ĵ���
        color_bs = frame['meta']['pre_fetched']['color']
        depth_bs = frame['meta']['pre_fetched']['depth']

        timer.tick('extract idx from mask')

        # ����batch
        bs = len(depth_bs)
        mask_bs_next = mask_bs.clone()
        # img_bs = color_bs.clone()
        img_bs = []
        choose_bs = []
        points_feat_bs = torch.tensor([]).cuda()  # ��ȡ�ļ�������
        points_bs = torch.tensor([]).cuda()  # (bs, 1024, 3)
        points_origin_bs = torch.tensor([]).cuda()
        timer_1 = Timer(True)
        for batch_idx in range(bs):
            color = color_bs[batch_idx]
            depth = depth_bs[batch_idx]
            mask = mask_bs[batch_idx]

            if self.remove_border_w > 0:
                mask = self.remove_border_bool(mask, kernel_size=self.remove_border_w)


            # ��ͨ��mask�õ�bbox��Ȼ�������ͼ����вü�
            if self.img_process == "zero_padding":
                # ����mask,����Χ���������
                color_zero_pad, bbox = zero_padding_by_mask(color, mask_bs[batch_idx])
                # img_bs[batch_idx] = color_zero_pad
                img_bs.append(color_zero_pad)
                timer_1.tick('single batch | zero_padding')
            if self.img_process == "crop":
                # ����mask,��RGB���вü�
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

            # ��mask���choose
            mask = torch.logical_and(mask, depth > 0)
            # choose��bbox�ü����mask���ɵģ����ǿ���ֱ��������ȡ�ü���ͼ���ϵ�����
            # �����Ҫ��ȡԭ��Сͼ���ϵ����ص㣬 ��������
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


            # 1.��ȡ��������

            # ����normal map�� ����bbox֮��
            if 'real' in self.mode:
                fx = 591.0125
                fy = 590.16775
            else:
                fx = 577.5
                fy = 577.5
            k_size = 3
            distance_threshold = 2000
            difference_threshold = 20  # ��Χ�ĵ���Ⱦ��볬��10mm�򲻿���
            point_into_surface = False
            # ʹ�òü�������ͼ����normal map
            depth_nor = depth[rmin:rmax, cmin:cmax].numpy().astype(np.uint16)
            normals_map_out = normalSpeed.depth_normal(depth_nor, fx, fy, k_size, distance_threshold, difference_threshold,
                                                       point_into_surface)
            timer.tick('depth 2 normal_end')
            nrm = normals_map_out

            # width_crop = depth[rmin:rmax, cmin:cmax].shape[1]
            # height_crop = depth[rmin:rmax, cmin:cmax].shape[0]


            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]       # �Ե��ƽ���һ������,����n_pts����
            # ������ȡ�ü�ǰͼ���ϵĵ㣬��Ҫ���ڷ�ͶӰ(��Ϊchoose���ڲü�ǰ��mask��2D bbox�ϵõ���)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]     # ��������u
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]     # ��������v

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

            # ������ȡ�ü���ͼ���ϵĵ㣬���ڴ�rgb��nrm����ȡ���ƶ�Ӧ������
            xmap_crop_masked = xmap_masked - cmin - 1
            ymap_crop_masked = ymap_masked - rmin - 1

            crop_w = rmax - rmin
            ratio = self.img_size / crop_w
            col_idx = choose % crop_w  # һά����ת�ü���Ķ�ά����
            row_idx = choose // crop_w
            # ����ü���Ķ�ά���굽resize��Ķ�ά���꣬Ȼ��תһά����
            choose_resize = (torch.floor(row_idx * ratio) * self.img_size + torch.floor(col_idx * ratio)).type(torch.int64)
            choose_bs.append(choose_resize)  # ���resizeͼ���Ӧ��choose (1024, 1)

            # points_viz = points.copy()  # ���Է�ͶӰ|(1)
            points = torch.from_numpy(np.transpose(points, (2, 0, 1))).cuda()  # points (1024, 3, 1) -> (1, 1024, 3)
            points = points.type(torch.float32)
            points_origin_bs = torch.cat((points_origin_bs, points), 0)  # δ��ֵ���ĵ���
            # ʹ�þ�ֵ���й�һ��
            # points_mean (10, 3, 1)
            # points (1, 1024, 3)
            points_mean_single_batch = torch.mean(points, axis=1, keepdims=True).cuda().repeat(1, points.shape[1], 1)
            # ���points_mean������batch��points��mean
            # points_mean_single_batch = frame['meta']['points_mean'][batch_idx].cuda().unsqueeze(0)
            points = points - points_mean_single_batch

            points_bs = torch.cat((points_bs, points), 0)  # ��ֵ���ĵ���
            timer_1.tick('single batch | backproject')

            # if self.mode == 'test':
            #     # ���Է�ͶӰ|(2)���ӻ�color �� mask
            #     print('data from {}'.format(frame['meta']['ori_path']))
            #     cv2.imshow('rgb', color.clone().cpu().numpy())
            #     cv2.waitKey(0)
            #     viz_mask_bool('viz_mask', mask)
            #     # ���Է�ͶӰ|(3)���ӻ��ü���rgb
            #     rgb_show = torch.zeros(color.shape, dtype=torch.uint8)
            #     rgb_show[ymap_masked, xmap_masked, 0] = color[ymap_masked, xmap_masked, 0]
            #     rgb_show[ymap_masked, xmap_masked, 1] = color[ymap_masked, xmap_masked, 1]
            #     rgb_show[ymap_masked, xmap_masked, 2] = color[ymap_masked, xmap_masked, 2]
            #     cv2.imshow('rgb', rgb_show.numpy())
            #     cv2.waitKey(0)
            #     # ���Է�ͶӰ|(3)���ӻ���ͶӰ����
            #     points_viz = np.squeeze(np.transpose(points_viz, (2, 0, 1)), 0)
            #     color_red = np.array([255, 0, 0])
            #     color_green = np.array([0, 255, 0])
            #     color_blue = np.array([0, 0, 255])
            #     pts_colors = [color_red]
            #     render_points_diff_color('test backproject', [points_viz],
            #                              pts_colors, save_img=False,
            #                              show_img=True)

            # img_cropped (3, 192, 192)
            # ��һ�����⣬img_cropped��ͨ��resize�ű��192x192�ģ�xmap_masked��ymap_maskedӦ����α任��
            # ������δ�ü���ͼ��+bbox+xmap�����
            # ��������������emb��������һ��ratio�ܷ���������⣿

            points_rgb = color[ymap_masked, xmap_masked, :]
            points_rgb = points_rgb.squeeze(2).transpose(0, 1).cuda()  # points_rgb -> (1, 1024, 3)
            points_nrm = nrm[ymap_crop_masked, xmap_crop_masked]
            points_nrm = torch.from_numpy(points_nrm).squeeze(1).transpose(1, 0).cuda()  # points_nrm (1024, 1, 1, 3) -> (1, 1024, 3)
            timer_1.tick('single batch | extract color and nrm feature')

            # ���Ҫ���Ӷ�������Ĺ��ˣ���Ҫ���²���(���¼���choose)�������ڼ���choose֮ǰ�ͽ��й���
            # viz_multi_points_diff_color(f'pts:{batch_idx}', [points], [np.array([[1, 0, 0]])])
            # ƴ��������ͬ������
            # �Ƿ��������nrm����ʧ������ʹ�ã�
            #  -> (1, 1024, 3) ->  -> (1, 1024, 9)
            pts_9d = torch.concat([points, points_rgb, points_nrm], dim=2)
            # ����֮ǰpoints_feat_bs �ǿյ�
            points_feat_bs = torch.cat((points_feat_bs, pts_9d), 0)  #######�Ա�SGPA��������Բ��У����� ���к���instance_geometryֻ������3ά
            timer_1.tick('single batch | concat 3 feat')


            # Ϊ��һ֡����mask_add
            mask_bs_next[batch_idx] = add_border_bool_by_crop_pos(mask, bbox, kernel_size=10)
            timer_1.tick('single batch | add border')
            # ����ֻ�м���ָ��Ӧ����

        timer.tick('go through batch and backproject pcd')
        # ��ȡ��������
        points_feat_bs = points_feat_bs.type(torch.float32)  # points_feat_bs (bs, 1024, 3)
        points_feat = self.instance_geometry(points_feat_bs[:, :, :3])  # points_feat (bs, 64, 1024)

        timer.tick('get geometry feature')
        points_rgb_feat = self.instance_rgbFeature(points_feat_bs[:, :, 3:6])  # (bs, 64, 1024)
        points_nrm_feat = self.instance_nrmFeature(points_feat_bs[:, :, 6:])  # (bs, 64, 1024)

        # ͼ������PSP-Net, ��ȡRGB����
        img_bs = torch.stack(img_bs, dim=0).cuda()  # convert list to tensor
        img_bs = img_bs.type(torch.float32)
        # SGPA��,�����ͼ���Ѿ����ü�Ϊ192x192��
        # ֮����Բ���һ��
        # problem: ��������ǿ��Լ����Ż��ģ���Ϊ����choose�Ὣû�õ�ɸѡ����������������ܶ�û��CNN

        # (bs, 3, 192, 192) -> (bs, 32, 192, 192)
        out_img = self.psp(img_bs)
        di = out_img.size()[1]  # ����ά�� 32
        emb = out_img.view(bs, di, -1)  # ��һ��ͼ����������ͼ����ÿ�����ص��
        timer.tick('get RGB feature CNN')

        # choose_bs list ת tensor
        choose_bs = torch.stack(choose_bs, dim=0).cuda()  # convert list to tensor
        choose_bs = choose_bs.type(torch.int64)

        # ����
        # ���������в���,����n_pts����
        # choose_bs (bs, n_pts=1024, 1) -> (bs, n_dim=32, n_pts=1024)
        # choose_bs = choose_bs.transpose(1, 2).repeat(1, di, 1)
        # ����choose��ȡ��Ӧ�������
        choose_bs = choose_bs.permute(0, 2, 1).repeat(1, di, 1)  # (bs=10, n_pts=1024, 1) -> (bs=10, di=32, n_pts=1024)
        emb = torch.gather(emb, 2, choose_bs).contiguous()
        # emb (bs, 64, n_pts)
        # emb (10, 64, 1024)
        # 1024�������ɫ������ÿ�����������64ά
        emb = self.instance_color(emb)
        timer.tick('get RGB feature sampling')

        # �����Ѿ��õ���(bs, 64, 1024)�ļ���������(bs, 64, 1024)����ɫ����
        # ������ƴ�ӵõ�instance local����
        # ���һ�¼�����������ɫ������ֵ��
        # ��ɫ����ʹ�þ�����
        inst_local = torch.cat((points_feat, emb), dim=1)   # bs x 128 x 1024

        # �������滻Ϊpointnet�Ľ��
        # inst_local = torch.cat((points_feat, points_rgb_feat), dim=1)   # bs x (64+64) x 1024 -> bs x 128 x 1024

        inst_global = self.instance_global(inst_local)      # bs x 1024 x 1
        timer.tick('get RGB feature concat')
        # ����Ҫ���ָ�mask_for_next
        #
        # kps_3d = None
        # mask_bs_next ����һ֡ʹ�õ�mask
        return inst_local, inst_global, mask_bs_next, points_bs, points_origin_bs

    def forward(self, data):
        # ��ȡ��Ҫ������,��ת�Ƶ�cpu,��ʹ���µ��ֵ����洢
        self.feed_dict = []
        # self.npcs_feed_dict = []
        for i, frame in enumerate(data):
            if i == 0:
                self.feed_dict.append(self.convert_init_frame_data(frame))
            else:
                self.feed_dict.append(self.convert_subseq_frame_data(frame))
        # �����data��Ϊ������, ��һ֡�ͺ���֡
        # 1.��ʼ֡
        # �Ƿ�������� if else
        init_frame = self.feed_dict[0]

        # ����mask�з�color
        init_pre_fetched = init_frame['meta']['pre_fetched']
        init_masks = init_pre_fetched['mask_add']
        init_colors = init_pre_fetched['color']
        batch_size = len(init_masks)
        init_crop_pos = []

        # try FFB6D extract 3D kp
        mask_last_frame = self.feed_dict[0]['meta']['pre_fetched']['mask']
        crop_pos_last_frame = init_crop_pos   # ��¼ÿ��ͼ����ĸ��ü�����

        points_assign_mat = []
        gt_pose_between_frame = []
        # ��i֡
        for i in range(1, len(self.feed_dict)):
            last_frame = self.feed_dict[i - 1]
            next_frame = self.feed_dict[1]
            last_colors = last_frame['meta']['pre_fetched']['color']
            next_colors = next_frame['meta']['pre_fetched']['color']
            # ��ȡ��֡��3D�ؼ���
            # ��ȡ��һ֡�Ĺؼ���
            print('extracting feature 1  ...')
            timer_extract_feat = Timer(True)
            inst_local_1, inst_global_1, mask_bs_next, points_bs_1, points_origin_bs_1 = self.extract_3D_kp(last_frame, mask_last_frame)
            timer_extract_feat.tick('extract feature 1 end')

            print('extracting feature 2  ...')


            # �����ã�ʹ�õڶ�֡��gt_mask�����Ҫ����ǰһ��ĵ�padding mask��ɾ����һ��
            mask_bs_next = self.feed_dict[i]['meta']['pre_fetched']['mask']



            inst_local_2, inst_global_2, _, points_bs_2, points_origin_bs_2 = self.extract_3D_kp(next_frame, mask_bs_next)
            timer_extract_feat.tick('extract feature 2 end')

            # �ο�SGPA�����Ӧ����A
            # 1 -> 2
            # pts1 = A1*pts2
            assign_matrix_bs_1 = self.get_assgin_matrix(inst_local_1, inst_global_1, inst_global_2)  # �ο�SGPA loss.py line 40: assign_matrix_bs_1 Ӧ����points_1���
            timer_extract_feat.tick('get_assgin_matrix 1 end')
            # 2 -> 1
            # pts2 = A2*pts1
            assign_matrix_bs_2 = self.get_assgin_matrix(inst_local_2, inst_global_2, inst_global_1) # assign_matrix_bs_1 Ӧ����points_1���
            timer_extract_feat.tick('get_assgin_matrix 2 end')

            points_assign_mat.append((points_bs_1, points_bs_2, assign_matrix_bs_1, assign_matrix_bs_2, points_origin_bs_1, points_origin_bs_2))
            # len(points_assign_mat)��֡������-1

            # ������֮֡���gtλ��
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
                # λ�˲����Ǵ�ģ������ϵ���۲���Ƶģ����Ǵ�nocs���۲���Ƶ�
                # ���Ӧ�õõ�nocs����
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

        # ���ģ��ͶӰ��ͼ����
        print(last_frame['meta']['path'])

        # ����λ�����
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
