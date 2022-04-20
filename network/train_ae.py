import os
device_ids = "2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
from data.dataset import NOCSDataset
from torch.utils.data import DataLoader
import torch
import cv2

def train():
    dataset_path = '/data1/cxx/Lab_work/dataset'
    result_path = '/data1/cxx/Lab_work/results'
    obj_category = '1'
    mode = 'train'
    num_expr = 'exp_tmp'
    device = torch.device("cuda:0")
    dataset = NOCSDataset(dataset_path=dataset_path,
                          result_path=result_path,
                          obj_category=obj_category,
                          mode=mode,
                          num_expr=num_expr,
                          device=device)
    print(f'Successfully Load NOCSDataSet {num_expr}_{mode}_{obj_category}')
    batch_size = 2
    total_epoch = 250
    shuffle = (mode == 'train')
    num_workers = 0

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    for i, data in enumerate(train_dataloader):
        print(f'data index {i}')
        print(data['path'])
        if 'real' in mode:
            # Real
            replace_str = '_composed'
        else:
            replace_str = '_depth'
        # 如何操作batch？？？？for？？？？看看captra和SPD是怎么做的
        depth = cv2.imread()
        # 读取深度图和mask， mask add_border， 提取normal map ， 自编码器训练


if __name__ == "__main__":
    train()


