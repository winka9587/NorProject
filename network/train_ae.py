# coding=utf-8
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='3, 4', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--lr_policy', type=str, default='step', help='')
parser.add_argument('--lr_step_size', type=str, default='step', help='')
parser.add_argument('--lr_gamma', type=str, default='step', help='')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
parser.add_argument('--result_dir', type=str, default='results/Real', help='directory to save train results')
parser.add_argument('--max_epoch', type=int, default=15, help='max number of epochs to train')
parser.add_argument('--log_path', type=str, default='../results/Real/', help='path to save tensorboard log file')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start(start count from 1)')
parser.add_argument('--CDLossTimes', type=float, default=1.0, help='')
parser.add_argument('--CorrLossTimes', type=float, default=1.0, help='')
parser.add_argument('--EntropyLossTimes', type=float, default=1.0, help='')
parser.add_argument('--device_ids', type=str, default="3", help='choose device to run code')
parser.add_argument('--log_close', type=bool, default='False', help='use log or not')
parser.add_argument('--exp_name', type=str, default='BioNet', help='name of this experiment')
parser.add_argument('--resume_model', type=str, default='', help='load model')


# 测试用
# parser.add_argument('--log_close', type=bool, default='True', help='name of this experiment')
# parser.add_argument('--exp_name', type=str, default='just test', help='name of this experiment')
# parser.add_argument('--resume_model', type=str, default='../results/Real/NOCSNoCDLoss-1.0-1.0-5.0-_model_cat1_04.pth', help='load model')

# backup
# parser.add_argument('--log_close', type=bool, default='True', help='name of this experiment')
# parser.add_argument('--exp_name', type=str, default='just test', help='name of this experiment')
# parser.add_argument('--resume_model', type=str, default='/data1/cxx/Lab_work/Exp/exp1/results/Real/BioDirectionHalfWeight_Regular20_model_cat1_25.pth', help='load model')
# parser.add_argument('--resume_model', type=str, default='../results/Real/TwoFrameSame_100RegularLoss_model_cat1_05.pth', help='load model')
# parser.add_argument('--dataset', type=str, default='CAMERA+Real', help='CAMERA or CAMERA+Real')
# parser.add_argument('--data_dir', type=str, default='data', help='data directory')
# parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
# parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
# parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
# parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
# parser.add_argument('--num_workers', type=int, default=8, help='number of data loading workers')
# parser.add_argument('--gpu', type=str, default='0,1', help='GPU to use')
# parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
# parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
# parser.add_argument('--max_epoch', type=int, default=15, help='max number of epochs to train')
# parser.add_argument('--resume_model', type=str, default='', help='resume from saved model')
# parser.add_argument('--result_dir', type=str, default='results/real', help='directory to save train results')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_ids

import cv2
import numpy as np
# device_ids = "3"
# os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
from data.dataset import RealSeqDataset
from torch.utils.data import DataLoader

import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(os.path.dirname(os.path.abspath(__file__)))
from network.only_sift import SIFT_Track
from lib.utils import pose_fit, render_points_diff_color
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from lib.loss import Loss
from tensorboardX import SummaryWriter


# lr_policy
# little utils
from utils import Timer
from tqdm import tqdm


def train(opt):
    # test_coord_correspondence2()
    # log
    writer = SummaryWriter(opt.log_path)
    if opt.log_close:
        writer.close()
    # Loss
    corr_wt = 0.5 * opt.CorrLossTimes   # 1.0
    cd_wt = 2.5 * opt.CDLossTimes  # 5.0
    entropy_wt = 0.00005 * opt.EntropyLossTimes  # 0.0001
    criterion = Loss(corr_wt, cd_wt, entropy_wt, writer)  # SPD 的loss

    dataset_path = '/data1/cxx/Lab_work/dataset'  # 数据集路径,格式like:CAPTRA
    result_path = '/data1/cxx/Lab_work/results'  # 保存数据集的预处理结果
    obj_category = '1'  # 类id, 当前模型仅针对该类进行训练
    mode = 'real_train'
    num_expr = '{}-{}-{}-{}-'.format(opt.exp_name, opt.CDLossTimes, opt.CorrLossTimes, opt.EntropyLossTimes)  # 实验编号

    # status output
    print(f'\n Exp start: {num_expr}\n')
    print(f'\n device id: {opt.device_ids}\n')
    print(f'\n Loss: \n corr: {corr_wt}\n CD: {cd_wt}\n Regular: {entropy_wt}\n')

    subseq_len = 2
    device = torch.device("cuda:0")
    train_dataset = RealSeqDataset(dataset_path=dataset_path,
                          result_path=result_path,
                          obj_category=obj_category,
                          mode=mode,
                          subseq_len=subseq_len,
                          num_expr=num_expr,
                          device=device)
    test_dataset = RealSeqDataset(dataset_path=dataset_path,
                                   result_path=result_path,
                                   obj_category=obj_category,
                                   mode='real_test',
                                   subseq_len=-1,
                                   num_expr=num_expr,
                                   device=device)
    print(f'Successfully Load NOCSDataSet {num_expr}_{mode}_{obj_category}')

    shuffle = (mode == 'train' or mode == 'real_train')  # 是否打乱
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    emb_dim = 512
    num_points = 1024


    # trainer = SIFT_Track(device=device, real=('real' in mode), mode='train', opt=opt, remove_border_w=-1, tb_writer=writer)
    trainer = SIFT_Track(device=device, real=('real' in mode), mode='train', opt=opt, remove_border_w=-1, tb_writer=writer)



    trainer.cuda()
    # trainer = nn.DataParallel(trainer)
    # trainer = torch.nn.DataParallel(trainer, device_ids=device_ids)

    # Optimizer
    decay_epoch = [0, 5, 10]
    decay_rate = [1.0, 0.6, 0.3]
    n_decays = len(decay_epoch)
    for i in range(n_decays):
        if opt.start_epoch > decay_epoch[i]:
            decay_count = i

    # origin optimizer without decay
    # optimizer = torch.optim.Adam(trainer.parameters(), lr=opt.lr)


    if opt.resume_model != '':
        trainer.load_state_dict(torch.load(opt.resume_model))
        print(f'load resume model from {opt.resume_model}')
    for epoch in tqdm(range(opt.start_epoch, opt.max_epoch + 1), desc='Epoch:', position=0, leave=True):
        for i, data in enumerate(tqdm(train_dataloader, desc='training:', position=0, leave=True)):

            # 测试eval用
            # if i == 0:
            #     break

            # test
            # if i < 100:
            #     continue

            if decay_count < len(decay_rate):
                if epoch > decay_epoch[decay_count]:
                    current_lr = opt.lr * decay_rate[decay_count]
                    optimizer = torch.optim.Adam(trainer.parameters(), lr=current_lr)
                    decay_count += 1

            message = {}
            message['exp_name'] = num_expr
            message['epoch'] = epoch
            message['step'] = i

            points_assign_mat, pose12, m1m2 = trainer(data)
            loss = criterion(points_assign_mat, pose12, m1m2, message)
            print('compute loss end')
            print(f'loss: {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # print(data['path'])
            # if 'real' in mode:
            #     # Real
            #     replace_str = '_composed'
            # else:
            #     replace_str = '_depth'
            # # 如何操作batch？？？？for？？？？看看captra和SPD是怎么做的
            # # depth = cv2.imread()
            # 读取深度图和mask， mask add_border， 提取normal map ， 自编码器训练

            # 在forward中 ,首先用mask add_border,然后裁剪depth，输入normalspeed
        # 保存模型
        model_save_path = os.path.join(os.path.abspath('..'), opt.result_dir)
        torch.save(trainer.state_dict(), '{0}/{3}_model_cat{1}_{2:02d}.pth'.format(model_save_path, obj_category, epoch, num_expr))
        print('train end, save model to {0}/{3}_model_cat{1}_{2:02d}.pth'.format(model_save_path, obj_category, epoch, num_expr))

        # # 测试
        # test_loss = {}
        # for i, data in enumerate(tqdm(test_dataloader, desc='testing', position=0, leave=True)):
        #     t2 = Timer(True)
        #     points_assign_mat_list, pose12 = trainer.test(data)
        #     t2.tick('predict end')
        #     # 评估位姿
        #     # points1和points2计算位姿
        #     total_loss = 0.0
        #     for frame_pair_idx in range(len(points_assign_mat_list)):
        #         frame_pair = points_assign_mat_list[frame_pair_idx]
        #         points_bs_1, points_bs_2, assign_matrix_bs_1, assign_matrix_bs_2 = frame_pair
        #         assign_matrix_bs_1 = F.softmax(assign_matrix_bs_1, dim=2)
        #         assign_matrix_bs_2 = F.softmax(assign_matrix_bs_2, dim=2)
        #
        #         assigned_points_1in2_bs = torch.bmm(assign_matrix_bs_1, points_bs_2)  # pts1在2坐标系下的映射
        #         # assigned_points_1in2_bs与point_bs_1计算位姿
        #         for j in range(len(assigned_points_1in2_bs)):
        #             assigned_points = assigned_points_1in2_bs[j].cpu().detach().numpy()
        #             points_1 = points_bs_1[j].cpu().detach().numpy()
        #             points_2 = points_bs_2[j].cpu().detach().numpy()
        #
        #             cv2.imshow('frame1 color', data[frame_pair_idx]['meta']['pre_fetched']['color'].numpy().squeeze(0))
        #             cv2.waitKey(0)
        #             cv2.imshow('frame2 color', data[frame_pair_idx + 1]['meta']['pre_fetched']['color'].numpy().squeeze(0))
        #             cv2.waitKey(0)
        #
        #             color_red = np.array([255, 0, 0])
        #             color_green = np.array([0, 255, 0])
        #             color_blue = np.array([0, 0, 255])
        #             pts_colors = [color_green, color_red]
        #             points_1 = points_bs_1[j].cpu().detach().numpy()
        #             render_points_diff_color('points_1:green points_2:red', [points_1, points_2],
        #                                      pts_colors, save_img=False,
        #                                      show_img=True)
        #             render_points_diff_color('assigned_points:green points_1:red', [assigned_points, points_1],
        #                                      pts_colors, save_img=False,
        #                                      show_img=True)
        #             render_points_diff_color('assigned_points:green points_2:red', [assigned_points, points_2],
        #                                      pts_colors, save_img=False,
        #                                      show_img=True)
        #             # 测试
        #             # points_1和2拟合位姿，然后变换1
        #             predicted_pose12 = pose_fit(points_1, assigned_points)  # 总是返回None是为什么？？？
        #             points_1_RT = np.matmul(predicted_pose12['rotation'], points_1.transpose()).transpose() * predicted_pose12[
        #                 'scale'] + predicted_pose12['translation'].transpose()
        #
        #             render_points_diff_color('pose_fit pts1:green pts2:red', [points_1_RT, points_2],
        #                                      pts_colors, save_img=False,
        #                                      show_img=True)
        #
        #
        #             predicted_pose12 = pose_fit(points_2, points_2)  # 模型预测的位姿
        #             # 获得前后两帧的sRT
        #             # meta中的位姿nocs2camera是 get_gt_poses.py中的函数
        #             # get_image_pose(num_instances, mask, coord, depth, intrinsics):
        #             # 通过函数 pose = pose_fit(coord_pts, pts)获得的
        #             sRT1 = data[frame_pair_idx]['meta']['nocs2camera'][0]
        #             sRT2 = data[frame_pair_idx+1]['meta']['nocs2camera'][0]
        #             R12 = torch.mm(torch.inverse(sRT1['rotation'].squeeze(0)), sRT2['rotation'].squeeze(0))
        #
        #             # 按理说 R12 应该与 gt meta中的points pose_fit得到的旋转一样
        #             gt_points1 = data[frame_pair_idx]['points'].cpu().squeeze(0).transpose(0, 1).numpy()
        #             gt_points2 = data[frame_pair_idx+1]['points'].cpu().squeeze(0).transpose(0, 1).numpy()
        #             gt_pose12 = pose_fit(gt_points1, gt_points2)
        #
        #
        #         # 与gt进行比较
        #
        #     #return total_loss




if __name__ == "__main__":
    train(opt)



