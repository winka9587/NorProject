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
                item = item.long().to(self.device)
            else:
                item = item.float().to(self.device)
            feed_frame[key] = item
        gt_part = part_model_batch_to_part(cvt_torch(frame['meta']['nocs2camera'], self.device), self.num_parts,
                                           self.device)
        feed_frame.update({'gt_part': gt_part})

        return feed_frame


    # ���շ��ص�input����:(��*�������init frame�������)
    # points    (B,3,4096)
    # meta      (B,3,1)
    # labels    (B,4096)
    # gt_part   (rotation,scale,translation)
    # points_mean *     (10,3,1)
    # npcs *            (10,3,4096)
    def convert_subseq_frame_data(self, data):
        # ����gtλ��
        gt_part = part_model_batch_to_part(cvt_torch(data['meta']['nocs2camera'], self.device), self.num_parts,
                                           self.device)
        # �۲����,�۲����ƽ����,gtλ��
        input = {'points': data['points'],
                 'points_mean': data['meta']['points_mean'],
                 'gt_part': gt_part}
        # ���nocs����
        if 'nocs' in data:
            input['npcs'] = data['nocs']
        input = cvt_torch(input, self.device)
        # ���meta
        input['meta'] = data['meta']
        # ���labels
        if 'labels' in data:
            input['labels'] = data['labels'].long().to(self.device)
        return input

    # ���յ�input����:
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


    # ��mask_add_from_last_frame˵���ǵڶ�֡����Ҫʹ��ǰһ֡�ṩ��mask
    def extract_3D_kp(self, frame, mask_bs):
        # rgb                ��ɫͼ��               [bs, 3, h, w]
        # dpt_nrm            ͼ��:xyz+normal       [bs, 6, h, w], 3c xyz in meter + 3c normal map
        # cld_rgb_nrm����:    xyz+rgb+normal      [bs, 9, npts]
        # choose             Ӧ����mask            [bs, 1, npts]

        # ����color,normal,depth,mask,��ͶӰ���õ�9ͨ���ĵ���
        color_bs = frame['meta']['pre_fetched']['color']
        depth_bs = frame['meta']['pre_fetched']['depth']
        nrm_bs = frame['meta']['pre_fetched']['nrm']

        # ����batch
        bs = len(depth_bs)
        mask_bs_next = mask_bs.clone()
        for batch_idx in range(bs):
            color = color_bs[batch_idx]
            depth = depth_bs[batch_idx]
            nrm = nrm_bs[batch_idx]
            mask = mask_bs[batch_idx]
            # 1.��ȡ��������
            # ����mask,��ͶӰ�õ��۲����
            points, idxs = backproject(depth, self.intrinsics, mask=mask_bs[batch_idx])
            points = torch.from_numpy(points)
            points_rgb = color[idxs[0], idxs[1]]
            points_nrm = nrm[idxs[0], idxs[1]]
            pts_9d = torch.concat([points, points_rgb, points_nrm], dim=1)


            # 2.��ȡ��ɫ����
            # ͨ��mask���crop_pos
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
                # ����idx[0] y
                # ����indx[1] x

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

            # ����mask,����Χ���������
            color_zero_pad = zero_padding_by_mask(color, mask_bs[batch_idx])
            # Ϊ��һ֡����mask_add
            mask_bs_next[batch_idx] = add_border_bool(mask, kernel_size=10)

            # ͼ������PSP-Net
            out_img = self.psp(color_zero_pad)
            di = out_img.size()[1]
            emb = out_img.view(bs, di, -1)
            choose = choose.unsqueeze(1).repeat(1, di, 1)
            emb = torch.gather(emb, 2, choose).contiguous()
            emb = self.instance_color(emb)




        # ����Ҫ���ָ�mask_for_next
        #
        kps_3d = None
        return kps_3d, mask_bs_next

    def forward(self):
        # �����data��Ϊ������, ��һ֡�ͺ���֡
        # 1.��ʼ֡
        # �Ƿ�������� if else
        init_frame = self.feed_dict[0]

        # ����mask�з�color
        init_pre_fetched = init_frame['meta']['pre_fetched']
        init_masks = init_pre_fetched['mask_add']
        init_colors = init_pre_fetched['color']
        init_nrms = init_pre_fetched['nrm']
        batch_size = len(init_masks)
        init_crop_pos = []
        # ���ԶԱȵĵ�:
        # last_frame ��mask
        # next_frame ��mask_add
        # for i in range(batch_size):
        #     idx = torch.where(init_masks[i, :, :])
        #     x_max = max(idx[1])
        #     x_min = min(idx[1])
        #     y_max = max(idx[0])
        #     y_min = min(idx[0])
        #     crop_pos_tmp = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
        #     init_crop_pos.append(crop_pos_tmp)
        use_ = False
        # ��ʱ��ʹ�õĴ���
        if use_:
            # sift
            for i in range(1, len(self.feed_dict)):
                last_frame = self.feed_dict[i - 1]
                next_frame = self.feed_dict[1]
                last_colors = last_frame['meta']['pre_fetched']['color']
                last_nrms = last_frame['meta']['pre_fetched']['nrm']
                next_colors = next_frame['meta']['pre_fetched']['color']
                next_nrms = next_frame['meta']['pre_fetched']['nrm']
                # siftƥ��
                for j in range(batch_size):
                    last_crop_color = crop_img(last_colors[j], crop_pos[j])
                    next_crop_color = crop_img(next_colors[j], crop_pos[j])
                    timer = Timer(True)
                    color_sift_1, kp_xys_1, des_1 = extract_sift_kp_from_RGB(last_crop_color)
                    color_sift_2, kp_xys_2, des_2 = extract_sift_kp_from_RGB(next_crop_color)
                    timer.tick('sift feature extract')
                    matches = sift_match(des_1, des_2, self.matcher)
                    timer.tick('sift match ')
                    # ������RANSAC����������
                    # https://blog.csdn.net/sinat_41686583/article/details/115186277

                    # ȡ��Ӧ��normal map
                    last_crop_nrm = crop_img(last_nrms[j], crop_pos[j])
                    next_crop_nrm = crop_img(next_nrms[j], crop_pos[j])
                    # ���ӻ�
                    # cv2.imshow('color_sift_1', color_sift_1)
                    # cv2.waitKey(0)
                    # cv2.imshow('color_sift_2', color_sift_2)
                    # cv2.waitKey(0)
                    #
                    # cv2.imshow('nrm_1', norm2bgr(last_crop_nrm))
                    # cv2.waitKey(0)
                    # cv2.imshow('nrm_2', norm2bgr(next_crop_nrm))
                    # cv2.waitKey(0)

                    # ��ȡ3D�㲢����ƥ��
                    # ��ȡxyz+RGB+normal��������ƥ��


        # try FFB6D extract 3D kp
        mask_last_frame = self.feed_dict[0]['meta']['pre_fetched']['mask']
        crop_pos_last_frame = init_crop_pos   # ��¼ÿ��ͼ����ĸ��ü�����
        # ��i֡
        for i in range(1, len(self.feed_dict)):
            last_frame = self.feed_dict[i - 1]
            next_frame = self.feed_dict[1]
            last_colors = last_frame['meta']['pre_fetched']['color']
            last_nrms = last_frame['meta']['pre_fetched']['nrm']
            next_colors = next_frame['meta']['pre_fetched']['color']
            next_nrms = next_frame['meta']['pre_fetched']['nrm']
            # ��ȡ��֡��3D�ؼ���
            # ��ȡ��һ֡�Ĺؼ���
            mask_add = self.extract_3D_kp(last_frame, mask_last_frame)
            self.extract_3D_kp(next_frame, mask_add)
            if i != len(self.feed_dict):
                # ���ڲ�ͬ��Сcrop CNN�Ĵ���
                # ��һ֡��ʹ��addborder(���Ƕ���add�Ĳ����Ƿ�Ҫ����Ϊ0��)
                # �ڶ�֡�õ�һ֡��addborder
                # �����λ�˺�,���¼���mask
                # ���к���֡
                # ����һ֡mask��ĵ���ͨ��RT�任���ڶ�֡,��Ϊ֮���֡�ṩmask
                mask_last_frame = get_mask()










        # bs = embedding.size()[0]
        # out = F.relu(self.fc1(embedding))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        # out_pc = out.view(bs, -1, 3)
        # return out_pc


    def set_data(self, data):
        # ��ȡ��Ҫ������,��ת�Ƶ�cpu,��ʹ���µ��ֵ����洢
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
        # ����forward,������loss
        # self.forward(save=save)
        # if not no_eval:
        #     self.compute_loss(test=True, per_instance=save, eval_iou=True, test_prefix='test')
        # else:
        #     self.loss_dict = {}
        pass


