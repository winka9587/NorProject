import os
import sys
import numpy as np
from typing import Tuple
from os.path import join as pjoin
from copy import deepcopy
import pickle
import glob
import cv2

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))
sys.path.append(os.path.join(base_dir, '..', '..'))

from captra_utils.utils_from_captra import backproject, project, get_corners, bbox_from_corners
from network.data_utils import farthest_point_sample
from utils import Timer

nocs_real_cam_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])


def subtract_mean(data):
    points_mean = np.mean(data['points'], axis=-1, keepdims=True)  # [3, N] -> [3, 1]
    data['points'] = data['points'] - points_mean
    data['meta']['points_mean'] = points_mean
    return data


def shuffle(data):
    n = data['points'].shape[-1]
    perm = np.random.permutation(n)  # 对0到n之间的数随机排列
    for key in data.keys():
        if key in ['meta', 'nocs_corners']:
            continue
        else:
            cur_perm = perm
        data[key] = data[key][..., cur_perm]
    return data


def add_corners(data, obj_info):
    corners = np.array(obj_info['corners'])
    nocs_corners = corners[1:].copy()
    nocs_corners /= np.sqrt(np.sum((nocs_corners[:, 1:] - nocs_corners[:, :1]) ** 2, axis=-1, keepdims=True))
    nocs_corners = nocs_corners - np.mean(nocs_corners, axis=1, keepdims=True)  # [P, 2, 3]
    data['meta']['nocs_corners'] = nocs_corners
    return data


# 读取render下的.npz文件中的点云和位姿
# 返回采样后的相机坐标系下点云, 对应mask点云, 扰动后的位姿
def read_cloud(cloud_dict, num_points, radius_factor, perturb_cfg, device):
    cam = cloud_dict['points']  # 相机坐标系下的点云 nx3, len(cam)的值为n
    if len(cam) == 0:
        return None, None
    seg = cloud_dict['labels']  # mask, nx1
    pose = deepcopy(cloud_dict['pose'])
    center = pose['translation'].reshape(3)  # 因为NOCS的中心就是(0,0,0),因此其加上位移t,就是对应到相机坐标系下的中心点
    scale = pose['scale']
    # 对t和s添加扰动
    if perturb_cfg is not None:
        center += random_translation(perturb_cfg['t'], (1,), perturb_cfg['type']).reshape(3)
        scale += random_vector(perturb_cfg['s'], (1,), perturb_cfg['type'])
    perturbed_pose = {'translation': center.reshape(pose['translation'].shape),
                      'scale': float(scale)}

    radius = float(scale * radius_factor)
    # 返回FPS采样结果的下标
    idx = crop_ball_from_pts(cam, center, radius, num_points=num_points, device=device)

    # 返回采样后的相机坐标系下点云, 对应mask点云, 扰动后的位姿
    return cam[idx], seg[idx], perturbed_pose


# 观测点云计算对应的nocs点云
def base_generate_data(cam_points, seg, pose):
    nocs = np.zeros_like(cam_points)
    idx = np.where(seg == 1)[0]
    nocs[idx] = np.matmul((cam_points[idx] - pose['translation'].swapaxes(-1, -2)) / pose['scale'],
                          pose['rotation'])
    full_data = {'points': cam_points, 'labels': 1 - seg, 'nocs': nocs,
                 'nocs2camera': [pose]}
    return full_data


def split_nocs_dataset(root_dset, obj_category, num_expr, mode, bad_ins=[]):
    # splits目录不存在,则创建该目录
    output_path = pjoin(root_dset, "splits", obj_category, num_expr)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if mode in ['real_test_can', 'real_test_bottle']:
        extra = mode[10:]
        dir = 'real_test'
    else:
        dir = mode
        extra = None
    extra_dict = {
        'bottle': ['shampoo_norm/scene_4'],
        'can': ['lotte']
    }
    # 在render/mode/obj_category/下存放的是该类的所有实例的数据
    path = pjoin(root_dset, "render", dir, obj_category)
    all_ins = [ins for ins in os.listdir(path) if not ins.startswith('.')]  # 存储该目录下的所有文件夹名
    data_list = []
    for instance in all_ins:
        # 跳过bad实例
        if instance in bad_ins:
            continue
        # path/instance/下的所有scene(test时instance下都是scene_n,train时instance下都是0000)，glob读取的是路径而不只是文件名
        # 读取的是一个实例模型出现的所有图片(场景)
        for track_dir in glob.glob(pjoin(path, instance, '*')):
            frames = glob.glob(pjoin(track_dir, 'data', '*'))  # scene下的所有的/data/文件
            cloud_list = [file for file in frames if file.endswith('.npz')]  # 读取scene_n/data中的.npz文件名
            cloud_list.sort(key=lambda str: int(str.split('.')[-2].split('/')[-1]))  # -2去掉后缀，-1去掉路径，只根据文件的数值进行排序
            data_list += cloud_list  # npz文件路径
    # 最终得到的路径类似于../render/train/1/109d55a137c042f5760315ac3bf2c13e/0000/data/00.npz
    # 将.npz文件路径写入到splits目录下的mode.txt中
    with open(pjoin(output_path, f'{mode}.txt'), 'w') as f:
        for item in data_list:
            if extra is not None:  # extra是做什么的？
                flag = False
                for keyword in extra_dict[extra]:
                    if keyword in item:
                        flag = True
                        break
                if not flag:
                    continue
            f.write('{}\n'.format(item))
    # 最终生成了real_test.txt文件


# 返回最远点采样的结果
def crop_ball_from_pts(pts, center, radius, num_points=None, device=None):
    # 计算所有点到中心点的距离
    # 相当于对冗余点进行了处理
    distance = np.sqrt(np.sum((pts - center) ** 2, axis=-1))  # [N]
    radius = max(radius, 0.05)  # 最大半径为0.05
    for i in range(10):
        # 以中心点为圆心,radius为半径画圆,选出包含的的点,一般情况下,默认的半径0.6会包含所有的点
        idx = np.where(distance <= radius)[0]
        # 如果包含的点数量少于10个,或者没有指定num_points,将半径增大为原来的1.1倍
        if len(idx) >= 10 or num_points is None:
            break
        radius *= 1.10

    if num_points is not None:
        # 当半径中包含的点数量大于10个时
        if len(idx) == 0:
            # 半径最大也没有包含点，则选择所有点
            idx = np.where(distance <= 1e9)[0]
        if len(idx) == 0:
            # 包含所有点也没有点，直接返回
            return idx
        # 点的数量少于采样数量,重复点
        while len(idx) < num_points:
            idx = np.concatenate([idx, idx], axis=0)
        fps_idx = farthest_point_sample(pts[idx], num_points, device)
        idx = idx[fps_idx]
    return idx


def random_vector(std, shape: Tuple, type='normal'):
    if type == 'normal':  # use std as std
        return np.random.randn(*shape) * std
    elif type == 'uniform':  # in range [-std, std]
        return np.random.rand(*shape) * 2 * std - std
    elif type == 'exact':  # return +/-std
        sign = np.random.randn(*shape)
        sign = sign / np.abs(sign)
        return sign * std
    else:
        assert 0, f'Unsupported random type {type}'


def random_translation(std, shape: Tuple, type='normal'):
    norm = random_vector(std, shape, type)
    direction = np.random.randn(*(shape + (3,)))
    direction = direction / np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
    jitter = norm * direction
    return jitter


def get_proj_corners(depth, center, radius,
                     cam_intrinsics=nocs_real_cam_intrinsics):
    radius = max(radius, 0.05)
    aa_corner = get_corners([center - np.ones(3) * radius * 1.0, center + np.ones(3) * radius * 1.0])
    aabb = bbox_from_corners(aa_corner)
    height, width = depth.shape
    projected_corners = project(aabb, cam_intrinsics).astype(np.int32)[:, [1, 0]]
    projected_corners[:, 0] = height - projected_corners[:, 0]
    corner_2d = np.stack([np.min(projected_corners, axis=0),
                          np.max(projected_corners, axis=0)], axis=0)
    corner_2d[0, :] = np.maximum(corner_2d[0, :], 0)
    corner_2d[1, :] = np.minimum(corner_2d[1, :], np.array([height - 1, width - 1]))
    return corner_2d


def crop_ball_from_depth_image(depth, mask, center, radius,
                               cam_intrinsics=nocs_real_cam_intrinsics,
                               num_points=None, device=None):
    corner_2d = get_proj_corners(depth, center, radius, cam_intrinsics)
    corner_mask = np.zeros_like(depth)
    corner_mask[corner_2d[0, 0]: corner_2d[1, 0] + 1, corner_2d[0, 1]: corner_2d[1, 1] + 1] = 1

    raw_pts, raw_idx = backproject(depth, intrinsics=cam_intrinsics, mask=corner_mask)
    raw_mask = mask[raw_idx[0], raw_idx[1]]

    idx = crop_ball_from_pts(raw_pts, center, radius, num_points, device=device)
    if len(idx) == 0:
        return crop_ball_from_depth_image(depth, mask, center, radius * 1.2, cam_intrinsics, num_points, device)
    pts = raw_pts[idx]
    obj_mask = raw_mask[idx]
    return pts, obj_mask


def compute_2d_bbox_iou(box, boxes):
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])

    def area(x1, x2, y1, y2):
        return np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    intersection = area(x1, x2, y1, y2)
    union = area(box[1], box[3], box[0], box[2]) + area(boxes[:, 1], boxes[:, 3], boxes[:, 0], boxes[:, 2]) - intersection
    iou = intersection / union
    return iou


def full_data_from_depth_image(depth_path, category, instance, center, radius, gt_pose, num_points=None, device=None,
                               mask_from_nocs2d=False, nocs2d_path=None,
                               pre_fetched=None):
    timer = Timer(False)
    if pre_fetched is None:
        depth = cv2.imread(depth_path, -1)
        timer.tick('Read depth image')
        with open(depth_path.replace('depth.png', 'meta.txt'), 'r') as f:
            meta_lines = f.readlines()
        inst_num = -1
        for meta_line in meta_lines:
            inst_num = int(meta_line.split()[0])
            inst_id = meta_line.split()[-1]
            if inst_id == instance:
                break
        timer.tick('Read meta file')

        mask = cv2.imread(depth_path.replace('depth', 'mask'))[:, :, 2]
        mask = (mask == inst_num)

        timer.tick('Read mask')
    else:
        depth, mask = pre_fetched['depth'].numpy().astype(np.uint16), pre_fetched['mask'].numpy()

    if mask_from_nocs2d:
        scene_name, frame_num = depth_path.split('/')[-2:]
        frame_num = frame_num[:4]
        nocs2d_result_path = pjoin(nocs2d_path, f'results_test_{scene_name}_{frame_num}.pkl')

        with open(nocs2d_result_path, 'rb') as f:
            nocs2d_result = pickle.load(f)

        pred_class_ids, pred_bboxes = nocs2d_result['pred_class_ids'], nocs2d_result['pred_bboxes']
        category = int(category)
        same_category = (pred_class_ids == category)
        if same_category.sum() == 0:
            print('no same class pred!', nocs2d_result_path)
        else:
            while True:
                track_bbox = get_proj_corners(depth, center, radius).reshape(-1)
                ious = compute_2d_bbox_iou(track_bbox, pred_bboxes)
                ious = ious * same_category
                if np.max(ious) > 0.05 or radius > 0.5:
                    break
                else:
                    radius *= 1.2
            best_pred = np.argmax(ious)
            mask = nocs2d_result['pred_masks'][..., best_pred]

    pts, obj_mask = crop_ball_from_depth_image(depth, mask, center, radius,
                                               num_points=num_points, device=device)
    timer.tick('crop ball')
    full_data = base_generate_data(pts, obj_mask, gt_pose)
    timer.tick('generate_full')
    return full_data



if __name__ == '__main__':
    print(random_vector(1.0, (2, 3), 'normal'))
    print(random_vector(1.0, (2, 3), 'uniform'))
    print(random_vector(1.0, (2, 3), 'exact'))
