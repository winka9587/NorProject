# coding=utf-8
import os
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
from lib.utils import pose_fit

parser = argparse.ArgumentParser()
# lr_policy

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
    val_dataset = RealSeqDataset(dataset_path=dataset_path,
                                   result_path=result_path,
                                   obj_category=obj_category,
                                   mode='test',
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
    test_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

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
                assigned_points2 = torch.bmm(assign_matrix_bs_2, points_bs_1)
                # assigned_points2与point_bs_2计算位姿
                for j in range(len(assigned_points2)):
                    assigned_points = assigned_points2[j].cpu().numpy()
                    points_2 = points_bs_2[j].cpu().numpy()
                    predicted_pose12 = pose_fit(assigned_points, points_2)  # 模型预测的位姿
                    # 获得前后两帧的sRT
                    # meta中的位姿nocs2camera是 get_gt_poses.py中的函数
                    # get_image_pose(num_instances, mask, coord, depth, intrinsics):
                    # 通过函数 pose = pose_fit(coord_pts, pts)获得的
                    sRT1 = data[frame_pair]['meta']['nocs2camera'][0]
                    sRT2 = data[frame_pair+1]['meta']['nocs2camera'][0]
                    R12 = torch.mm(torch.inverse(sRT1['rotation'].squeeze(0)), sRT2['rotation'].squeeze(0))

                    # 按理说 R12 应该与 gt meta中的points pose_fit得到的旋转一样
                    gt_points1 = data[frame_pair]['points'].cpu().squeeze(0).transpose(0, 1).numpy()
                    gt_points2 = data[frame_pair+1]['points'].cpu().squeeze(0).transpose(0, 1).numpy()
                    gt_pose12 = pose_fit(gt_points1, gt_points2)


                # 与gt进行比较
                print(pose)
                print(data)
            return total_loss

        # 保存模型
        torch.save(trainer.state_dict(), '{0}/model_cat{1}_{2:02d}.pth'.format(opt.result_dir, obj_category, epoch))


if __name__ == "__main__":
    opt = parser.parse_args()
    train(opt)



