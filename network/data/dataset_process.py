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
    perm = np.random.permutation(n)  # ��0��n֮������������
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


# ��ȡrender�µ�.npz�ļ��еĵ��ƺ�λ��
# ���ز�������������ϵ�µ���, ��Ӧmask����, �Ŷ����λ��
def read_cloud(cloud_dict, num_points, radius_factor, perturb_cfg, device):
    cam = cloud_dict['points']  # �������ϵ�µĵ��� nx3, len(cam)��ֵΪn
    if len(cam) == 0:
        return None, None
    seg = cloud_dict['labels']  # mask, nx1
    pose = deepcopy(cloud_dict['pose'])
    center = pose['translation'].reshape(3)  # ��ΪNOCS�����ľ���(0,0,0),��������λ��t,���Ƕ�Ӧ���������ϵ�µ����ĵ�
    scale = pose['scale']
    # ��t��s����Ŷ�
    if perturb_cfg is not None:
        center += random_translation(perturb_cfg['t'], (1,), perturb_cfg['type']).reshape(3)
        scale += random_vector(perturb_cfg['s'], (1,), perturb_cfg['type'])
    perturbed_pose = {'translation': center.reshape(pose['translation'].shape),
                      'scale': float(scale)}

    radius = float(scale * radius_factor)
    # ����FPS����������±�
    idx = crop_ball_from_pts(cam, center, radius, num_points=num_points, device=device)

    # ���ز�������������ϵ�µ���, ��Ӧmask����, �Ŷ����λ��
    return cam[idx], seg[idx], perturbed_pose


# �۲���Ƽ����Ӧ��nocs����
def base_generate_data(cam_points, seg, pose):
    nocs = np.zeros_like(cam_points)
    idx = np.where(seg == 1)[0]
    nocs[idx] = np.matmul((cam_points[idx] - pose['translation'].swapaxes(-1, -2)) / pose['scale'],
                          pose['rotation'])
    full_data = {'points': cam_points, 'labels': 1 - seg, 'nocs': nocs,
                 'nocs2camera': [pose]}
    return full_data


def split_nocs_dataset(root_dset, obj_category, num_expr, mode, bad_ins=[]):
    # splitsĿ¼������,�򴴽���Ŀ¼
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
    # ��render/mode/obj_category/�´�ŵ��Ǹ��������ʵ��������
    path = pjoin(root_dset, "render", dir, obj_category)
    all_ins = [ins for ins in os.listdir(path) if not ins.startswith('.')]  # �洢��Ŀ¼�µ������ļ�����
    data_list = []
    for instance in all_ins:
        # ����badʵ��
        if instance in bad_ins:
            continue
        # path/instance/�µ�����scene(testʱinstance�¶���scene_n,trainʱinstance�¶���0000)��glob��ȡ����·������ֻ���ļ���
        # ��ȡ����һ��ʵ��ģ�ͳ��ֵ�����ͼƬ(����)
        for track_dir in glob.glob(pjoin(path, instance, '*')):
            frames = glob.glob(pjoin(track_dir, 'data', '*'))  # scene�µ����е�/data/�ļ�
            cloud_list = [file for file in frames if file.endswith('.npz')]  # ��ȡscene_n/data�е�.npz�ļ���
            cloud_list.sort(key=lambda str: int(str.split('.')[-2].split('/')[-1]))  # -2ȥ����׺��-1ȥ��·����ֻ�����ļ�����ֵ��������
            data_list += cloud_list  # npz�ļ�·��
    # ���յõ���·��������../render/train/1/109d55a137c042f5760315ac3bf2c13e/0000/data/00.npz
    # ��.npz�ļ�·��д�뵽splitsĿ¼�µ�mode.txt��
    with open(pjoin(output_path, f'{mode}.txt'), 'w') as f:
        for item in data_list:
            if extra is not None:  # extra����ʲô�ģ�
                flag = False
                for keyword in extra_dict[extra]:
                    if keyword in item:
                        flag = True
                        break
                if not flag:
                    continue
            f.write('{}\n'.format(item))
    # ����������real_test.txt�ļ�


# ������Զ������Ľ��
def crop_ball_from_pts(pts, center, radius, num_points=None, device=None):
    # �������е㵽���ĵ�ľ���
    # �൱�ڶ����������˴���
    distance = np.sqrt(np.sum((pts - center) ** 2, axis=-1))  # [N]
    radius = max(radius, 0.05)  # ���뾶Ϊ0.05
    for i in range(10):
        # �����ĵ�ΪԲ��,radiusΪ�뾶��Բ,ѡ�������ĵĵ�,һ�������,Ĭ�ϵİ뾶0.6��������еĵ�
        idx = np.where(distance <= radius)[0]
        # ��������ĵ���������10��,����û��ָ��num_points,���뾶����Ϊԭ����1.1��
        if len(idx) >= 10 or num_points is None:
            break
        radius *= 1.10

    if num_points is not None:
        # ���뾶�а����ĵ���������10��ʱ
        if len(idx) == 0:
            # �뾶���Ҳû�а����㣬��ѡ�����е�
            idx = np.where(distance <= 1e9)[0]
        if len(idx) == 0:
            # �������е�Ҳû�е㣬ֱ�ӷ���
            return idx
        # ����������ڲ�������,�ظ���
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
