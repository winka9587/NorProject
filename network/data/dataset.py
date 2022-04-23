# coding=utf-8
import math
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
# # 指定显卡
# device_ids = "2,3"
# os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
import torch
from torch.utils.data import Dataset, DataLoader

from os.path import join as pjoin
import cv2
from utils import ensure_dir
from captra_utils.utils_from_captra import load_depth
import _pickle as cPickle
from dataset_process import split_nocs_dataset, read_cloud, base_generate_data, shuffle, subtract_mean, add_corners
from utils import add_border_bool, Timer
import normalSpeed
import random
from copy import deepcopy


# 读取实例模型对应的npy文件
def generate_nocs_corners(dataset_path, instance):
    obj_path = pjoin(dataset_path, 'model_corners', f'{instance}.npy')
    nocs_corners = np.load(obj_path).reshape(1, -1, 3)  # shape (2, 3) -> (1, 2, 3)
    return nocs_corners


def generate_nocs_data(root_dset, mode, obj_category, instance, track_num, frame_i, num_points, radius, perturb_cfg,
                       device):
    # instance      109d55a137c042f5760315ac3bf2c13e
    # track_num     0000
    # _,            data
    # frame_i:      00
    # path: ../dataset/render/train/1/109d55a137c042f5760315ac3bf2c13e/0000
    path = pjoin(root_dset, 'render', mode, obj_category, instance, f'{track_num}')
    # path to load .npz
    cloud_path = pjoin(path, 'data', f'{frame_i}.npz')
    cloud_dict = np.load(cloud_path, allow_pickle=True)['all_dict'].item()  # ndarray.item()仅适用于有一个元素的数组，将其转换为python的标量返回
    # cloud_dict = {points: (n, 3)
    #               labels: (n, )
    #               pose: {rotation, scale, translation}
    #               path: /data1/cxx/Lab_work/dataset/nocs_full/train/00045/0006_composed.png
    # }
    # 返回采样后的相机坐标系下点云, 对应mask点云, 扰动后的位姿
    cam_points, seg, perturbed_pose = read_cloud(cloud_dict, num_points, radius, perturb_cfg, device)
    if cam_points is None:
        return None
    # 传入的是未扰动的gt位姿
    full_data = base_generate_data(cam_points, seg, cloud_dict['pose'])
    full_data['ori_path'] = cloud_dict['path']  # 将路径和扰动位姿也存入字典
    full_data['crop_pose'] = [perturbed_pose]
    # real数据集还要额外读取深度图和mask(单个实例的)
    if 'real' in mode:
        depth_path = cloud_dict['path']
        depth = cv2.imread(depth_path, -1)
        with open(depth_path.replace('depth.png', 'meta.txt'), 'r') as f:
            meta_lines = f.readlines()
        inst_num = -1
        find_inst = False
        for meta_line in meta_lines:
            inst_num = int(meta_line.split()[0])
            inst_id = meta_line.split()[-1]
            if inst_id == instance:
                find_inst = True
                break
        if not find_inst:
            print(f'Error when trying find instance in meta.txt, no such instance {instance} in scene {depth_path}')
        mask = cv2.imread(depth_path.replace('depth', 'mask'))[:, :, 2]
        mask = (mask == inst_num)
        full_data['pre_fetched'] = {'depth': depth.astype(np.int16), 'mask': mask}
    else:
        # 不同于CAPTRA, CAMERA数据集也需要深度图和mask
        depth_path = cloud_dict['path']
        depth = cv2.imread(depth_path, -1)
        with open(depth_path.replace('composed.png', 'meta.txt'), 'r') as f:
            meta_lines = f.readlines()
        inst_num = -1
        find_inst = False
        for meta_line in meta_lines:
            inst_num = int(meta_line.split()[0])
            inst_id = meta_line.split()[-1]
            if inst_id == instance:
                find_inst = True
                break
        if not find_inst:
            print(f'Error when trying find instance in meta.txt, no such instance {instance} in scene {depth_path}')
        mask = cv2.imread(depth_path.replace('depth', 'mask'))[:, :, 2]
        mask = (mask == inst_num)
        full_data['pre_fetched'] = {'depth': depth.astype(np.int16), 'mask': mask}
        # origin CAPTRA code
        # full_data['pre_fetched'] = {}

    #
    # crop img by mask
    mask_add = add_border_bool(mask, kernel_size=10)
    # mask获取2D bbox,切割rgb图像
    idx = np.where(mask_add == True)
    # mask几乎没有点, 数据不可用
    if len(idx[0]) < 10:
        print(f"mask point not enough, instance {instance} in scene {depth_path}")
        return None
    # x_max = max(idx[1])
    # x_min = min(idx[1])
    # y_max = max(idx[0])
    # y_min = min(idx[0])
    # print(f'x:[{x_min}:{x_max}], y[{y_min}:{y_max}]')
    # depth_cropped = depth[y_min:y_max + 1, x_min:x_max + 1]
    # full_data['pre_fetched']['depth_cropped'] = depth_cropped.astype(np.int16)
    # full_data['pre_fetched']['crop_x_min'] = x_min
    # full_data['pre_fetched']['crop_y_min'] = y_min

    # 因为存储不同尺寸的图像会导致batchsize的问题，可以将裁剪留到网络里
    # 如果必须存储不同尺寸的图像，可以修改dataloader的参数
    # https://www.jianshu.com/p/da3bf988c246
    full_data['pre_fetched']['mask_add'] = mask_add


    # 计算normalSpeed
    # 遇到一个问题，就是normalspeed在计算时需要内参，其内部是做了反投影或类似的事情，但因为是封装好的，所以没法修改成处理某一
    # Real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # CAMERA_intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])
    fx = 577.5
    fy = 577.5

    k_size = 5
    distance_threshold = 2000
    difference_threshold = 20
    point_into_surface = False

    # color = np.load("examplePicture/color.npy")
    depth_nor = depth.astype(np.uint16)
    """
    The coordinate of depth and normal is in cv coordinate:
        - x is horizontal
        - y is down (to align to the actual pixel coordinates used in digital images)
        - right-handed: positive z look-at direction
    """
    timer = Timer(True)
    normals_map_out = normalSpeed.depth_normal(depth_nor, fx, fy, k_size, distance_threshold, difference_threshold,
                                               point_into_surface)
    timer.tick('depth_normal_end')
    full_data['pre_fetched']['nrm'] = normals_map_out
    # try viz nrm
    # cv2.imshow(f"normal map of {depth_path}", normals_map_out)
    # cv2.waitKey(0)

    # full_data: {
    # 'points': cam_points,
    # 'labels': 1 - seg,
    # 'nocs': nocs,
    # 'nocs2camera': [pose]
    # 'ori_path': cloud_dict['path']  # 将路径和扰动位姿也存入字典
    # 'crop_pose': [perturbed_pose]
    # 'pre_fetched': {'depth': depth.astype(np.int16), 'mask': mask}
    # }
    return full_data


def read_nocs_pose(root_dset, mode, obj_category, instance, track_num, frame_i):

    path = pjoin(root_dset, 'render', mode, obj_category, instance, f'{track_num}')
    cloud_path = pjoin(path, 'data', f'{frame_i}.npz')
    cloud_dict = np.load(cloud_path, allow_pickle=True)['all_dict'].item()
    pose = cloud_dict['pose']
    return pose


# result_path:
#       --img_list
# obj_category: [1, 2, 3, 4, 5, 6]
# mode: [train, val, real_train, real_val]
# source: [Real, CAMERA, Real_CAMERA]
class NOCSDataset(Dataset):
    def __init__(self, dataset_path, result_path, obj_category, mode, num_expr, num_points=4096, radius=0.6,
                 bad_ins=[], perturb_cfg=None, device=None, opt=None):
        print('Initializing NOCSDataset ...')
        # assert (mode == 'train' or mode == 'val'), 'Mode must be "train" or "val".'
        self.dataset_path = dataset_path
        self.result_path = result_path
        self.obj_category = obj_category
        self.mode = mode
        self.num_expr = num_expr  # 一个名字，区分不同的实验的数据
        self.num_points = num_points
        self.radius = radius
        self.perturb_cfg = perturb_cfg
        self.device = device
        self.opt = opt
        self.bad_ins = bad_ins  # bad_ins是一个数组,存储实例模型名，将不好的实例从数据集file_list中剔除
        self.file_list = self.collect_data()
        self.len = len(self.file_list)
        self.nocs_corner_dict = {}  # instance x (1, 2, 3)
        self.invalid_dict = {}      # 存储invalid的下标
        print('Successfully Initialized NOCSDataset ...')


    def collect_data(self):
        # ../data/nocs_data/splits/1/1_bottle_rot
        splits_path = pjoin(self.dataset_path, "splits", self.obj_category, self.num_expr)
        idx_txt = pjoin(splits_path, f'{self.mode}.txt')

        # 如果没有{mode}.txt，使用split_nocs_dataset函数生成一个
        splits_ready = os.path.exists(idx_txt)
        if not splits_ready:
            print(f'generating nocs dataset of category {self.num_expr}_{self.obj_category}_{self.mode} ...')
            split_nocs_dataset(self.dataset_path, self.obj_category, self.num_expr, self.mode, self.bad_ins)
        # 读取mode.txt文件
        print(f'NOCSDataSet loading category {self.obj_category} .npz file path from {idx_txt}')
        with open(idx_txt, "r", errors='replace') as fp:
            lines = fp.readlines()
            file_list = [line.strip() for line in lines]  # 将每一行的npz文件路径存入file_list中
        # if downsampling is not None:
        #     file_list = file_list[::downsampling]
        #
        # if truncate_length is not None:
        #     file_list = file_list[:truncate_length]
        return file_list

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        print(f'NOCSDataset get index {index}')
        # ../render/train/1/109d55a137c042f5760315ac3bf2c13e/0000/data/00.npz
        path = self.file_list[index]
        # ../render/train/1/109d55a137c042f5760315ac3bf2c13e/0000/data/00
        # ///
        last_name_list = path.split('.')[-2].split('/')
        # instance      109d55a137c042f5760315ac3bf2c13e
        # track_num     0000
        # _,            data
        # frame_i:      00
        instance, track_num, _, frame_i = last_name_list[-4:]  # 可以看出 instance就是具体到哪个实例
        fake_last_name = '/'.join(last_name_list[:-2] + last_name_list[-1:])  # 将[-2]也就是data丢弃掉
        # fake_last_name:   ../render/train/1/109d55a137c042f5760315ac3bf2c13e/0000/00
        fake_path = f'{fake_last_name}.pkl'  # 在和data同级目录创建一个pkl文件

        if instance not in self.nocs_corner_dict:
            # 如果当前的instance不在nocs_corner_dict中,读取该实例对应的npy文件
            self.nocs_corner_dict[instance] = generate_nocs_corners(self.dataset_path, instance)
        # 对不在黑名单中的实例
        if index not in self.invalid_dict:
            full_data = generate_nocs_data(self.dataset_path, self.mode, self.obj_category, instance,
                                           track_num, frame_i, self.num_points, self.radius,
                                           self.perturb_cfg, self.device)
            # full_data: {
            # 'points': cam_points,
            # 'labels': 1 - seg,
            # 'nocs': nocs,
            # 'nocs2camera': [pose]
            # 'ori_path': cloud_dict['path']  # 将路径和扰动位姿也存入字典
            # 'crop_pose': [perturbed_pose]
            # 'pre_fetched': {'depth': depth.astype(np.int16), 'mask': mask}
            # }
            if full_data is None:
                self.invalid_dict[index] = True
        if index in self.invalid_dict:
            return self.__getitem__((index + 1) % self.len)


        nocs2camera = full_data.pop('nocs2camera')
        crop_pose = full_data.pop('crop_pose')
        ori_path = full_data.pop('ori_path')
        pre_fetched = full_data.pop('pre_fetched')
        return {'data': full_data,
                'meta': {'path': fake_path,             # .npz路径
                         'ori_path': ori_path,          # png路径
                         'nocs2camera': nocs2camera,    # gt位姿
                         'crop_pose': crop_pose,        # 扰动位姿
                         'pre_fetched': pre_fetched,    # real 数据集 depth和mask, CAMERA数据集为None
                         'nocs_corners': self.nocs_corner_dict[instance]    # (1, 2, 3) bbox？角点
                         }
                }


# 返回file_list中不同路径的数量，存放在start_points中（为什么要用个list？直接用个数不行吗）
# 因為file_list中有多個scene的路徑，最終返回的数组其实是每个scene数据的起点
def get_seq_file_list_index(file_list):
    track_dict = {}
    start_points = []
    for i, file_name in enumerate(file_list):
        # 这一步唯一的作用是将最后的文件名删除了 ../xxx/xxx/0001.npz 变为 ../xxx/xxx
        track_name = '/'.join(file_name.split('/')[:-1])
        # 创建了一个字典，key是../xxx/xxx，value是文件在文件夹中sort后的序号
        # 字典中的key是npz文件的路径，i是对应的index从0开始，最后将所有的index以start_points
        if track_name not in track_dict:
            track_dict[track_name] = i
            start_points.append(i)
    print(track_dict)
    start_points.append(len(file_list))
    return start_points


class SequenceDataset(Dataset):
    # 加载数据集在PointData里面做了
    def __init__(self, dataset_path, result_path, obj_category, mode, num_expr, num_points=4096, radius=0.6,
                 bad_ins=[], perturb_cfg=None, device=None, opt=None):
        self.dataset = NOCSDataset(dataset_path, result_path, obj_category, mode, num_expr,
                                              num_points=num_points, radius=radius, bad_ins=bad_ins,
                                              perturb_cfg=perturb_cfg, device=device, opt=opt)
        self.seq_start = get_seq_file_list_index(self.dataset.file_list)
        print('seq start', self.seq_start)
        self.len = len(self.seq_start) - 1

    def __len__(self):
        return self.len

    def retrieve_single_frame(self, item):
        data = self.dataset[item]
        data_dict = data['data']
        meta_dict = data['meta']

        def reshape(x):  # from [N, x, x] to [C, N]
            x = x.reshape(x.shape[0], -1)
            x = x.swapaxes(0, 1)
            return x

        for key in data_dict.keys():
            if key in ['labels']:
                continue
            data_dict[key] = reshape(data_dict[key])
        data_dict['meta'] = meta_dict

        item_idx = data['meta']['path'].split('.')[-2].split('/')[-3]
        if 'nocs_corners' not in data_dict['meta'] and not self.real_data:
            data_dict = add_corners(data_dict, self.ins_info[item_idx])

        return data_dict

    def __getitem__(self, idx):
        seq_data = []
        for i in range(self.seq_start[idx], self.seq_start[idx + 1]):
            data = deepcopy(self.retrieve_single_frame(i))
            data = shuffle(data)
            seq_data.append(data)
        ret = []
        for data in seq_data:
            data = subtract_mean(data)
            ret.append(data)
        return ret


if __name__ == "__main__":
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
        print(data)



