# coding=utf-8
import os
import cv2
import numpy as np
device_ids = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
from data.dataset import RealSeqDataset
from torch.utils.data import DataLoader
import argparse
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

def test_coord_correspondence():
    # 测试coord图能否找到对应点
    # 在两幅coord图上连线对应点, xy1 tuple(x, y)
    def coord_line(img, img_p, xy1, xy2_s):
        # img_ = np.hstack((coord1, coord2))
        img_ = img.copy()
        img_p = img_p.copy()
        for xy2 in xy2_s:
            y2 = xy2[1]
            x2 = xy2[0]
            # x2 += img.shape[1]
            x2 += int(img.shape[1] / 2)
            cv2.line(img_, xy1, (x2, y2), (255, 0, 0), thickness=1)
            img_p[xy1[1], xy1[0], :] = 0
            img_p[xy1[1], xy1[0], 2] = 255
            img_p[y2, x2, :] = 0
            img_p[y2, x2, 2] = 255
        # cv2.imshow('coord respondence', img_)
        # cv2.waitKey(0)
        return img_, img_p

    prefix_1 = '0000'
    prefix_2 = '0040'
    target_instance_id_1 = 1  # 去找meta文件，想要哪个模型，就取其第一个值
    target_instance_id_2 = 5
    coord_1_path = f'/data1/cxx/Lab_work/dataset/nocs_full/real_train/scene_1/{prefix_1}_coord.png'
    coord_2_path = f'/data1/cxx/Lab_work/dataset/nocs_full/real_train/scene_1/{prefix_2}_coord.png'
    coord_1 = cv2.imread(coord_1_path)
    coord_2 = cv2.imread(coord_2_path)
    mask_1 = cv2.imread(coord_1_path.replace('coord', 'mask'))[:, :, 2]
    mask_2 = cv2.imread(coord_2_path.replace('coord', 'mask'))[:, :, 2]
    mask_1 = np.where(mask_1 == target_instance_id_1, True, False)
    mask_2 = np.where(mask_2 == target_instance_id_2, True, False)

    # coord_1 = data[0]['meta']['pre_fetched']['coord'].squeeze(0).numpy()
    # coord_2 = data[1]['meta']['pre_fetched']['coord'].squeeze(0).numpy()
    # mask_1 = data[0]['meta']['pre_fetched']['mask'].squeeze(0).numpy()
    # mask_2 = data[1]['meta']['pre_fetched']['mask'].squeeze(0).numpy()
    idx1 = np.where(mask_1 == False)
    idx2 = np.where(mask_2 == False)
    coord_1[idx1] = 0
    coord_2[idx2] = 0
    coord_c = np.hstack((coord_1, coord_2))
    coord_p = coord_c.copy()
    idx_non_zero_1 = coord_1.nonzero()
    color_0 = coord_2[:, :, 0].flatten()
    color_1 = coord_2[:, :, 1].flatten()
    color_2 = coord_2[:, :, 2].flatten()
    for idx_1 in range(len(idx_non_zero_1[0])):
        y1 = idx_non_zero_1[0][idx_1]
        x1 = idx_non_zero_1[1][idx_1]
        xy2_s = []
        # 遍历coord_1中的点,寻找早coord2中的对应点
        color = coord_1[y1, x1, :]
        print(f'searching {color}')
        _idx1 = np.where(color_0 == color[0])
        _idx2 = np.where(color_1 == color[1])
        _idx3 = np.where(color_2 == color[2])
        result_idx = reduce(np.intersect1d, [_idx1, _idx2, _idx3])
        if result_idx.shape[0] > 0:
            for e_idx in range(result_idx.shape[0]):
                idx_in_flatten = result_idx[e_idx]
                y = int(idx_in_flatten / 640)
                x = idx_in_flatten % 640
                print(f'idx:{e_idx}; pos:({y},{x}); value:{coord_2[y, x, :]} '
                      f'value2:{[color_0[idx_in_flatten], color_1[idx_in_flatten], color_2[idx_in_flatten]]}')
                xy2_s.append((x, y))
            coord_c = np.hstack((coord_1, coord_2))
            coord_c, coord_p = coord_line(coord_c, coord_p, (x1, y1), xy2_s)  # 可视化对应关系
    cv2.imshow('points', coord_p)
    cv2.waitKey(0)

# 测试新的求corr方法
def test_coord_correspondence2():
    # 测试coord图能否找到对应点
    # 在两幅coord图上连线对应点, xy1 tuple(x, y)
    def coord_line(img, img_p, xy1, xy2_s):
        # img_ = np.hstack((coord1, coord2))
        img_ = img.copy()
        img_p = img_p.copy()
        for xy2 in xy2_s:
            y2 = xy2[1]
            x2 = xy2[0]
            # x2 += img.shape[1]
            x2 += int(img.shape[1] / 2)
            cv2.line(img_, xy1, (x2, y2), (255, 0, 0), thickness=1)
            img_p[xy1[1], xy1[0], :] = 0
            img_p[xy1[1], xy1[0], 2] = 255
            img_p[y2, x2, :] = 0
            img_p[y2, x2, 2] = 255
        # cv2.imshow('coord respondence', img_)
        # cv2.waitKey(0)
        return img_, img_p

    # 在两张coord图上寻找颜色相同的点
    # coord已经用mask处理过
    def find_coord_correspondence(coord_1, coord_2):
        t = Timer(True)
        coord_1_flatten = coord_1.flatten().reshape(coord_1.shape[0] * coord_1.shape[1], -1)
        coord_2_flatten = coord_2.flatten().reshape(coord_2.shape[0] * coord_2.shape[1], -1)
        t.tick('flatten and reshape')
        # 能否在这里的时候将r和c一起append进去来节省时间？对应（1）的位置需要修改匹配
        coord_1_flatten_set = {(r, g, b) for [r, g, b] in coord_1_flatten if [r, g, b] != [0, 0, 0]}
        coord_2_flatten_set = {(r, g, b) for [r, g, b] in coord_2_flatten if [r, g, b] != [0, 0, 0]}
        t.tick('to set')
        coord_intersect_color = coord_1_flatten_set.intersection(coord_2_flatten_set)  # 取交集
        t.tick('intersection')
        corr_list = []
        for r, g, b in coord_intersect_color:
            # （1）这里也得修改
            r1, c1 = np.where((coord_1 == [r, g, b]).all(axis=-1))
            r2, c2 = np.where((coord_2 == [r, g, b]).all(axis=-1))
            assert len(r1) == 1 and len(c1) == 1 and len(r2) == 1 and len(c2) == 1
            corr_list.append((r1[0], c1[0], r2[0], c2[0]))
        t.tick('append to list')
        return corr_list

    # 返回的idx对应coord_1的坐标,可以直接通过coord_1[r,c]来进行读取
    def find_coord_correspondence2(coord_1, coord_2):
        print('==========new method==========')
        hash_1 = coord_1.copy()
        hash_2 = coord_2.copy()

        t = Timer(True)
        coord_1_flatten = coord_1.flatten().reshape(coord_1.shape[0] * coord_1.shape[1], -1)
        coord_2_flatten = coord_2.flatten().reshape(coord_2.shape[0] * coord_2.shape[1], -1)
        t.tick('flatten and reshape')
        # 能否在这里的时候将r和c一起append进去来节省时间？对应（1）的位置需要修改匹配
        # unique会丢失idx
        coord_1_flatten_set, idx_1 = np.unique(coord_1_flatten, return_index=True, axis=0)
        coord_2_flatten_set, idx_2 = np.unique(coord_2_flatten, return_index=True, axis=0)
        zero_idx1 = np.where((coord_1_flatten_set == [0, 0, 0]).all(axis=-1))[0][0]
        zero_idx2 = np.where((coord_2_flatten_set == [0, 0, 0]).all(axis=-1))[0][0]
        coord_1_flatten_set = np.delete(coord_1_flatten_set, [0, 0, 0], axis=0)
        coord_2_flatten_set = np.delete(coord_2_flatten_set, [0, 0, 0], axis=0)
        idx_1 = np.delete(idx_1, idx_1[zero_idx1], axis=0)
        idx_2 = np.delete(idx_2, idx_2[zero_idx2], axis=0)

        # coord_1_flatten_set = np.delete(, '[0, 0, 0]\n', axis=0)
        # coord_2_flatten_set = np.delete(np.unique(coord_2_flatten, axis=0), '[0, 0, 0]\n', axis=0)
        # 只能用set.intersection处理{tuple}
        # 不能用np.intersect1d来处理ndarray，即使ndarray中存的是tuple也不行，当ndarray超过一维，会被flatten
        # 一个方法，为每个color计算一个哈希值
        coord_1_flatten_set = [str(c) for c in coord_1_flatten_set]
        coord_2_flatten_set = [str(c) for c in coord_2_flatten_set]

        color_s, f_idx_1s, f_idx_2s = np.intersect1d(coord_1_flatten_set, coord_2_flatten_set, True, True)  # 取交集
        f_idx_1s = idx_1[f_idx_1s]
        f_idx_2s = idx_2[f_idx_2s]

        t.tick('intersection')
        corr_list = []
        w1 = coord_1.shape[1]
        w2 = coord_2.shape[1]
        for i in range(len(color_s)):
            r1 = int(f_idx_1s[i] / w1)
            c1 = f_idx_1s[i] % w1
            r2 = int(f_idx_2s[i] / w2)
            c2 = f_idx_2s[i] % w2
            corr_list.append((r1, c1, r2, c2))
        t.tick('append to list')
        return corr_list

    # 读取输入
    prefix_1 = '0000'
    prefix_2 = '0020'
    target_instance_id_1 = 1  # 去找meta文件，想要哪个模型，就取其第一个值
    target_instance_id_2 = 4
    coord_1_path = f'/data1/cxx/Lab_work/dataset/nocs_full/real_train/scene_1/{prefix_1}_coord.png'
    coord_2_path = f'/data1/cxx/Lab_work/dataset/nocs_full/real_train/scene_1/{prefix_2}_coord.png'
    coord_1 = cv2.imread(coord_1_path)
    coord_2 = cv2.imread(coord_2_path)
    mask_1 = cv2.imread(coord_1_path.replace('coord', 'mask'))[:, :, 2]
    mask_2 = cv2.imread(coord_2_path.replace('coord', 'mask'))[:, :, 2]
    mask_1 = np.where(mask_1 == target_instance_id_1, True, False)
    mask_2 = np.where(mask_2 == target_instance_id_2, True, False)

    # coord_1 = data[0]['meta']['pre_fetched']['coord'].squeeze(0).numpy()
    # coord_2 = data[1]['meta']['pre_fetched']['coord'].squeeze(0).numpy()
    # mask_1 = data[0]['meta']['pre_fetched']['mask'].squeeze(0).numpy()
    # mask_2 = data[1]['meta']['pre_fetched']['mask'].squeeze(0).numpy()
    idx1 = np.where(mask_1 == False)
    idx2 = np.where(mask_2 == False)
    coord_1[idx1] = 0
    coord_2[idx2] = 0

    # new
    timer = Timer(True)
    corr_list1 = find_coord_correspondence(coord_1, coord_2)
    print(f'len(corr_list): {len(corr_list1)}')
    timer.tick('find correspondence of two coord old method')
    corr_list2 = find_coord_correspondence2(coord_1, coord_2)
    print(f'len(corr_list): {len(corr_list2)}')
    timer.tick('find correspondence of two coord new method')
    coord_p0 = np.hstack((coord_1, coord_2))
    for r1, c1, r2, c2 in corr_list2:
        c2 += int(coord_1.shape[1])
        coord_p0[r1, c1, :] = 0
        coord_p0[r1, c1, 2] = 255
        coord_p0[r2, c2, :] = 0
        coord_p0[r2, c2, 2] = 255
    # cv2.imshow('new method', coord_p0)
    # cv2.waitKey(0)

    # old
    coord_c = np.hstack((coord_1, coord_2))
    coord_p = coord_c.copy()
    idx_non_zero_1 = coord_1.nonzero()
    color_0 = coord_2[:, :, 0].flatten()
    color_1 = coord_2[:, :, 1].flatten()
    color_2 = coord_2[:, :, 2].flatten()
    for idx_1 in range(len(idx_non_zero_1[0])):
        y1 = idx_non_zero_1[0][idx_1]
        x1 = idx_non_zero_1[1][idx_1]
        xy2_s = []
        # 遍历coord_1中的点,寻找早coord2中的对应点
        color = coord_1[y1, x1, :]
        # print(f'searching {color}')
        _idx1 = np.where(color_0 == color[0])
        _idx2 = np.where(color_1 == color[1])
        _idx3 = np.where(color_2 == color[2])
        result_idx = reduce(np.intersect1d, [_idx1, _idx2, _idx3])
        if result_idx.shape[0] > 0:
            for e_idx in range(result_idx.shape[0]):
                idx_in_flatten = result_idx[e_idx]
                y = int(idx_in_flatten / 640)
                x = idx_in_flatten % 640
                # print(f'idx:{e_idx}; pos:({y},{x}); value:{coord_2[y, x, :]} '
                #      f'value2:{[color_0[idx_in_flatten], color_1[idx_in_flatten], color_2[idx_in_flatten]]}')
                xy2_s.append((x, y))
            coord_c = np.hstack((coord_1, coord_2))
            # print('start draw')
            coord_c, coord_p = coord_line(coord_c, coord_p, (x1, y1), xy2_s)  # 可视化对应关系
    # cv2.imshow('points', coord_p)
    # cv2.waitKey(0)

    tmp = coord_p0 - coord_p
    print(f'if {len(np.where(tmp==0)[0])} == {coord_p.shape[0]*coord_p.shape[1]*coord_p.shape[2]} ? ')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='0, 1, 2, 3', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--lr_policy', type=str, default='step', help='')
parser.add_argument('--lr_step_size', type=str, default='step', help='')
parser.add_argument('--lr_gamma', type=str, default='step', help='')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
parser.add_argument('--result_dir', type=str, default='results/Real', help='directory to save train results')
parser.add_argument('--max_epoch', type=int, default=25, help='max number of epochs to train')
parser.add_argument('--log_path', type=str, default='../results/Real/', help='path to save tensorboard log file')
parser.add_argument('--start_epoch', type=int, default=11, help='which epoch to start')
# parser.add_argument('--log_close', type=bool, default='False', help='name of this experiment')
# parser.add_argument('--exp_name', type=str, default='PreTrain_UpRegularLoss', help='name of this experiment')
# parser.add_argument('--resume_model', type=str, default='../results/Real/PreTrain_UpRegularLoss_model_cat1_10.pth', help='load model')
# 测试用
parser.add_argument('--log_close', type=bool, default='True', help='name of this experiment')
parser.add_argument('--exp_name', type=str, default='just test', help='name of this experiment')
parser.add_argument('--resume_model', type=str, default='../results/Real/Only_CDLoss1_CorrLoss1_model_cat1_25.pth', help='load model')

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



def train(opt):
    # test_coord_correspondence2()

    dataset_path = '/data1/cxx/Lab_work/dataset' # 数据集路径,格式like:CAPTRA
    result_path = '/data1/cxx/Lab_work/results'  # 保存数据集的预处理结果
    obj_category = '1'  # 类id, 当前模型仅针对该类进行训练
    mode = 'real_train'
    num_expr = opt.exp_name  # 实验编号
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

    writer = SummaryWriter(opt.log_path)
    if opt.log_close:
        writer.close()
    shuffle = (mode == 'train' or mode == 'real_train')  # 是否打乱
    # shuffle = False
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    emb_dim = 512
    num_points = 1024


    # trainer = SIFT_Track(device=device, real=('real' in mode), mode='train', opt=opt, remove_border_w=-1, tb_writer=writer)
    trainer = SIFT_Track(device=device, real=('real' in mode), mode='train', opt=opt, remove_border_w=-1, tb_writer=writer)
    # Loss
    corr_wt = 1.0  # 1.0
    cd_wt = 5.0  # 5.0
    entropy_wt = 0.0005  # 0.0001
    criterion = Loss(corr_wt, cd_wt, entropy_wt, writer)  # SPD 的loss


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
    opt = parser.parse_args()
    train(opt)



