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
from network.viz_normal import SIFT_Track_normal_viz
from lib.utils import pose_fit, render_points_diff_color

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
    # test_dataset = RealSeqDataset(dataset_path=dataset_path,
    #                                result_path=result_path,
    #                                obj_category=obj_category,
    #                                mode='real_test',
    #                                subseq_len=-1,
    #                                num_expr=num_expr,
    #                                device=device)
    print(f'Successfully Load NOCSDataSet {num_expr}_{mode}_{obj_category}')


    batch_size = 10
    total_epoch = 250
    shuffle = (mode == 'train')  # 是否打乱
    shuffle = False
    num_workers = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    emb_dim = 512
    num_points = 1024
    resume_model = ''
    # resume_model = 'results/real/model_cat1_15.pth'

    trainer = SIFT_Track_normal_viz(device=device, real=('real' in mode), mode='train', opt=opt)
    # trainer = torch.nn.DataParallel(trainer, device_ids)
    trainer.cuda()
    if resume_model != '':
        trainer.load_state_dict(torch.load(resume_model))
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):
        print(f'epoch:{epoch}')
        for i, data in enumerate(train_dataloader):
            # 测试eval用
            if i == 1:
                break
            print(f'data index {i}')
            trainer.set_data(data)
            trainer.update()

            # 测试，得到两帧的mask与nrm与depth
            # 反投影得到点云xyz，
            def test_NCD(nrm1, nrm2, mask1, mask2):
                # 一、mask+nrm -> 法向nrm的 nx3矩阵
                idx1 = torch.where(mask1)
                idx2 = torch.where(mask2)
                nrm_pcd1 = nrm1[idx1]
                nrm_pcd2 = nrm2[idx2]

                # 二、两组nx3计算NCD


            nrm1 = data[0]['meta']['pre_fetched']['nrm'][0]  # (480, 640, 3)
            nrm2 = data[1]['meta']['pre_fetched']['nrm'][0]
            mask1 = data[0]['meta']['pre_fetched']['mask'][0]
            mask2 = data[1]['meta']['pre_fetched']['mask'][0]
            test_NCD(nrm1, nrm2, mask1, mask2)

if __name__ == "__main__":
    opt = parser.parse_args()
    train(opt)



