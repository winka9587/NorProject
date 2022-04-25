import torch
import torch.nn as nn
import torch.nn.functional as F
from part_dof_utils import part_model_batch_to_part, eval_part_full, add_noise_to_part_dof, \
    compute_parts_delta_pose
from utils import cvt_torch, Timer
from network.lib.utils import crop_img
import numpy as np
from extract_2D_kp import extract_sift_kp_from_RGB, sift_match

class SIFT_Track(nn.Module):
    def __init__(self, device, subseq_len=2):
        super(SIFT_Track, self).__init__()
        # self.fc1 = nn.Linear(emb_dim, 512)
        # self.fc2 = nn.Linear(512, 1024)
        # self.fc3 = nn.Linear(1024, 3*n_pts)
        self.subseq_len = subseq_len
        self.device = device
        self.num_parts = 1
        self.num_joints = 0

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


    def forward(self):
        # 传入的data分为两部分, 第一帧和后续帧
        # 1.初始帧
        # 是否添加噪声 if else
        init_frame = self.feed_dict[0]

        # 根据mask切分color
        init_pre_fetched = init_frame['meta']['pre_fetched']
        init_masks = init_pre_fetched['mask_add']
        init_colors = init_pre_fetched['color']
        batch_size = len(init_masks)
        crop_pos = []
        for i in range(batch_size):
            idx = torch.where(init_masks[i, :, :])
            x_max = max(idx[1])
            x_min = min(idx[1])
            y_max = max(idx[0])
            y_min = min(idx[0])
            crop_pos_tmp = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
            crop_pos.append(crop_pos_tmp)
        # sift
        for i in range(1, len(self.feed_dict)):
            if i == 1:
                last_frame = init_frame
                last_colors = init_colors
            next_frame = self.feed_dict[1]
            next_pre_fetched = next_frame['meta']['pre_fetched']
            next_colors = next_pre_fetched['color']
            # sift匹配
            bf = cv2.BFMatcher(cv2.NORM_L2)
            for j in batch_size:
                last_crop_color = crop_img(last_colors[j], crop_pos[j])
                next_crop_color = crop_img(next_colors[j], crop_pos[j])
                timer = Timer(True)
                color_sift_1, kp_xys_1, des_1 = extract_sift_kp_from_RGB(last_crop_color)
                color_sift_2, kp_xys_2, des_2 = extract_sift_kp_from_RGB(next_crop_color)
                Timer.tick('sift feature extract')
                matches = sift_match(des_1,des_2)
                Timer.tick('sift match ')
            last_frame = next_frame
            last_colors = next_colors
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


