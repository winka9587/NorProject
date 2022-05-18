# coding=utf-8
import os
import cv2
import numpy as np
device_ids = "4"
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
import torch.nn.functional as F
from functools import reduce
parser = argparse.ArgumentParser()
# lr_policy


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


parser.add_argument('--lr_policy', type=str, default='step', help='')
parser.add_argument('--lr_step_size', type=str, default='step', help='')
parser.add_argument('--lr_gamma', type=str, default='step', help='')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--max_epoch', type=int, default=15, help='max number of epochs to train')
parser.add_argument('--result_dir', type=str, default='results/Real', help='directory to save train results')

# parser.add_argument('--dataset', type=str, default='CAMERA+Real', help='CAMERA or CAMERA+Real')
# parser.add_argument('--data_dir', type=str, default='data', help='data directory')
# parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
# parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
# parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
# parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
# parser.add_argument('--batch_size', type=int, default=24, help='batch size')
# parser.add_argument('--num_workers', type=int, default=8, help='number of data loading workers')
# parser.add_argument('--gpu', type=str, default='0,1', help='GPU to use')
# parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
# parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
# parser.add_argument('--max_epoch', type=int, default=15, help='max number of epochs to train')
# parser.add_argument('--resume_model', type=str, default='', help='resume from saved model')
# parser.add_argument('--result_dir', type=str, default='results/real', help='directory to save train results')



def train(opt):
    test_coord_correspondence()

    dataset_path = '/data1/cxx/Lab_work/dataset' # 数据集路径,格式like:CAPTRA
    result_path = '/data1/cxx/Lab_work/results'  # 保存数据集的预处理结果
    obj_category = '1'  # 类id, 当前模型仅针对该类进行训练
    mode = 'real_train'
    num_expr = 'exp_tmp'  # 实验编号
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


    batch_size = 10
    total_epoch = 250
    shuffle = (mode == 'train')  # 是否打乱
    shuffle = False
    num_workers = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    emb_dim = 512
    num_points = 1024
    resume_model = ''
    # resume_model = 'results/real/model_cat1_15.pth'

    trainer = SIFT_Track(device=device, real=('real' in mode), mode='train', opt=opt)
    # trainer = torch.nn.DataParallel(trainer, device_ids)
    trainer.cuda()
    if resume_model != '':
        trainer.load_state_dict(torch.load(resume_model))
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):
        print(f'epoch:{epoch}')
        for i, data in enumerate(train_dataloader):
            # 测试eval用
            if i == 0:
                break
            print(f'data index {i}')
            trainer.set_data(data)
            trainer.update()

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
        print('train end')
        test_loss = {}
        for i, data in enumerate(test_dataloader):

            points_assign_mat_list = trainer.test(data)
            # 评估位姿
            # points1和points2计算位姿
            total_loss = 0.0
            for frame_pair_idx in range(len(points_assign_mat_list)):
                frame_pair = points_assign_mat_list[frame_pair_idx]
                points_bs_1, points_bs_2, assign_matrix_bs_1, assign_matrix_bs_2 = frame_pair
                assign_matrix_bs_1 = F.softmax(assign_matrix_bs_1, dim=2)
                assign_matrix_bs_2 = F.softmax(assign_matrix_bs_2, dim=2)

                assigned_points2 = torch.bmm(assign_matrix_bs_2, points_bs_1)
                # assigned_points2与point_bs_2计算位姿
                for j in range(len(assigned_points2)):
                    assigned_points = assigned_points2[j].cpu().detach().numpy()
                    points_2 = points_bs_2[j].cpu().detach().numpy()

                    cv2.imshow('frame1 color', data[frame_pair_idx]['meta']['pre_fetched']['color'].numpy().squeeze(0))
                    cv2.waitKey(0)
                    cv2.imshow('frame2 color', data[frame_pair_idx + 1]['meta']['pre_fetched']['color'].numpy().squeeze(0))
                    cv2.waitKey(0)

                    color_red = np.array([255, 0, 0])
                    color_green = np.array([0, 255, 0])
                    color_blue = np.array([0, 0, 255])
                    pts_colors = [color_green, color_red]
                    points_1 = points_bs_1[j].cpu().detach().numpy()
                    render_points_diff_color('points_1:green points_2:red', [points_1, points_2],
                                             pts_colors, save_img=False,
                                             show_img=True)
                    render_points_diff_color('assigned_points:green points_1:red', [assigned_points, points_1],
                                             pts_colors, save_img=False,
                                             show_img=True)
                    render_points_diff_color('assigned_points:green points_2:red', [assigned_points, points_2],
                                             pts_colors, save_img=False,
                                             show_img=True)
                    # 测试
                    # points_1和2拟合位姿，然后变换1
                    predicted_pose12 = pose_fit(assigned_points, points_2)  # 总是返回None是为什么？？？
                    points_1_RT = np.matmul(predicted_pose12['rotation'], points_1.transpose()).transpose() * predicted_pose12[
                        'scale'] + predicted_pose12['translation'].transpose()
                    render_points_diff_color('pose_fit pts1:green pts2:red', [points_1_RT, points_2],
                                             pts_colors, save_img=False,
                                             show_img=True)


                    predicted_pose12 = pose_fit(points_2, points_2)  # 模型预测的位姿
                    # 获得前后两帧的sRT
                    # meta中的位姿nocs2camera是 get_gt_poses.py中的函数
                    # get_image_pose(num_instances, mask, coord, depth, intrinsics):
                    # 通过函数 pose = pose_fit(coord_pts, pts)获得的
                    sRT1 = data[frame_pair_idx]['meta']['nocs2camera'][0]
                    sRT2 = data[frame_pair_idx+1]['meta']['nocs2camera'][0]
                    R12 = torch.mm(torch.inverse(sRT1['rotation'].squeeze(0)), sRT2['rotation'].squeeze(0))

                    # 按理说 R12 应该与 gt meta中的points pose_fit得到的旋转一样
                    gt_points1 = data[frame_pair_idx]['points'].cpu().squeeze(0).transpose(0, 1).numpy()
                    gt_points2 = data[frame_pair_idx+1]['points'].cpu().squeeze(0).transpose(0, 1).numpy()
                    gt_pose12 = pose_fit(gt_points1, gt_points2)


                # 与gt进行比较

            return total_loss

        # 保存模型
        torch.save(trainer.state_dict(), '{0}/model_cat{1}_{2:02d}.pth'.format(opt.result_dir, obj_category, epoch))


if __name__ == "__main__":
    opt = parser.parse_args()
    train(opt)



