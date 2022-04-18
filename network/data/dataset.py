import h5py
import numpy as np
import torch.utils.data as data
from os.path import join as pjoin
from utils import ensure_dir

# result_path:
# result_path:
#       --img_list
# mode: [train, test]
# source: [Real, CAMERA, Real_CAMERA]
class NOCSDataset(data.Dataset):
    def __init__(self, dataset_path, result_path, obj_category, mode, source, n_points=4096, augment=False):
        assert (mode == 'train' or mode == 'val'), 'Mode must be "train" or "val".'
        self.dataset_path = dataset_path
        self.result_path = result_path
        self.obj_category = obj_category
        self.mode = mode
        self.n_points = n_points
        self.augment = augment

        # augmentation parameters
        self.sigma = 0.01
        self.clip = 0.02
        self.shift_range = 0.02

        # img_list
        img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
                           'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']
        # ����ʹ�õ����ݼ���ѡ���_path��ɾȥ��Щtxt��pkl�ļ�
        if mode == 'train':
            del img_list_path[2:]
            del model_file_path[2:]
        else:
            del img_list_path[:2]
            del model_file_path[:2]

        if source == 'CAMERA':
            del img_list_path[-1]
            del model_file_path[-1]
        elif source == 'Real':
            del img_list_path[0]
            del model_file_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del img_list_path[0]
                del model_file_path[0]

    # ��·����ȡ���ݼ�
    def collect_data(self, truncate_length, downsampling):
        # ../data/nocs_data/splits/1/1_bottle_rot
        splits_path = pjoin(self.root_dset, "splits", self.obj_category, self.num_expr)
        idx_txt = pjoin(splits_path, f'{self.mode}.txt')
        print(f'����NOCSDataSet load data from {idx_txt}')
        # ���û��{mode}.txt��ʹ��split_nocs_dataset��������һ��
        splits_ready = os.path.exists(idx_txt)
        if not splits_ready:
            split_nocs_dataset(self.root_dset, self.obj_category, self.num_expr, self.mode, self.bad_ins)
        # ��ȡmode.txt�ļ�
        with open(idx_txt, "r", errors='replace') as fp:
            lines = fp.readlines()
            file_list = [line.strip() for line in lines]  # ��ÿһ�е�npz�ļ�·������file_list��

        if downsampling is not None:
            file_list = file_list[::downsampling]

        if truncate_length is not None:
            file_list = file_list[:truncate_length]

        return file_list

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # mytest
        # print(f'getitem from ShapeDataset {self.mode}')
        xyz = self.data[index]  # ��ShapeNetCore_4096.h5Ϊ�����õ���xyz��4096x3
        label = self.label[index] - 1    # data saved indexed from 1
        # print(f'[origin] xyz shape = {xyz.shape}')
        # print(f'[origin] label = {label}')
        # randomly downsample
        np_data = xyz.shape[0]
        assert np_data >= self.n_points, 'Not enough points in shape.'
        # ��xyz���������
        idx = np.random.choice(np_data, self.n_points)  # np_data��xyz�ĳ��ȣ������������n_points��ֵ������Щֵ��Ϊ�±���ȡ����ֵ
        # print(f'random sampling {self.n_points} points from {np_data} points')
        xyz = xyz[idx, :]
        # data augmentation
        if self.augment:
            jitter = np.clip(self.sigma*np.random.randn(self.n_points, 3), -self.clip, self.clip)
            xyz[:, :3] += jitter
            shift = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
            xyz[:, :3] += shift
        # print(f'xyz shape = {xyz.shape}')
        # print(f'label = {label}')
        return xyz, label
