# coding=utf-8
import os
device_ids = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
from data.dataset import RealSeqDataset
from torch.utils.data import DataLoader

import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(os.path.dirname(os.path.abspath(__file__)))
from network.only_sift import SIFT_Track

def train():
    dataset_path = '/data1/cxx/Lab_work/dataset'
    result_path = '/data1/cxx/Lab_work/results'
    obj_category = '1'
    mode = 'real_train'
    num_expr = 'exp_tmp'
    subseq_len = 2
    device = torch.device("cuda:0")
    dataset = RealSeqDataset(dataset_path=dataset_path,
                          result_path=result_path,
                          obj_category=obj_category,
                          mode=mode,
                          subseq_len=subseq_len,
                          num_expr=num_expr,
                          device=device)
    print(f'Successfully Load NOCSDataSet {num_expr}_{mode}_{obj_category}')


    batch_size = 10
    total_epoch = 250
    shuffle = (mode == 'train')  # 是否打乱
    shuffle = False
    num_workers = 0
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    emb_dim = 512
    num_points = 1024
    resume_model = ''
    estimator = SIFT_Track(device=device, real=('real' in mode))
    # estimator = torch.nn.DataParallel(estimator, device_ids)
    estimator.cuda()
    if resume_model != '':
        estimator.load_state_dict(torch.load(resume_model))

    for i, data in enumerate(train_dataloader):
        print(f'data index {i}')
        estimator.set_data(data)
        estimator.update()
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


if __name__ == "__main__":
    train()


