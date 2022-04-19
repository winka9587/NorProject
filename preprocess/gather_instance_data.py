import os
import sys

import numpy as np
import cv2
import pickle
from os.path import join as pjoin
import argparse
from multiprocessing import Process

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..', '..'))

from utils import ensure_dirs
from captra_utils.utils_from_captra import backproject, project, get_corners, bbox_from_corners
from tqdm import tqdm


def gather_instances(list_path, data_path, model_path, output_path, instances, intrinsics,
                     flip=True, real=False):
    # 对每一个实例,收集其所有数据
    for instance in tqdm(instances):
        gather_instance(list_path, data_path, model_path, output_path,
                        instance, intrinsics, flip=flip, real=real)


def gather_instance(list_path, data_path, model_path, output_path, instance, intrinsics,
                    flip=True, real=False, img_per_folder=100, render_rgb=False):
    # 读取/nocs_data/model_corners/instance_id.npy
    # 获得包围盒的8个顶点
    corners = np.load(pjoin(model_path, f'{instance}.npy'))
    bbox = bbox_from_corners(corners)
    bbox *= 1.4  # 包围盒大小乘以1.4
    # 获得有实例模型instance的所有路径
    meta_path = pjoin(list_path, f'{instance}.txt')
    with open(meta_path, 'r') as f:
        lines = f.readlines()

    # nocs_data/instance_data/(instance_id)
    inst_output_path = pjoin(output_path, instance)

    if not real:
        # 如果不是真实数据集,创建路径nocs_data/instance_data/(instance_id)/0000
        folder_num, img_num = 0, -1
        cur_folder_path = pjoin(inst_output_path, f'{folder_num:04d}')
        # render_rgb = False
        # nocs_data/instance_data/(instance_id)/0000/data
        # 如果render_rgb则再创建一个rgb目录
        ensure_dirs([pjoin(cur_folder_path, name) for name in (['data'] if not render_rgb else ['rgb', 'data'])])

    meta_dict = {}

    # (instance_id).txt中的每一行line是一个路径,例如00000/0001
    # 指向train/00000/0001_color.png
    for line in tqdm(lines, desc=f'Instance {instance}'):
        # 00000/0001
        # 00000 是track_name
        # 0001  是prefix
        track_name, prefix = line.strip().split('/')[:2]
        file_path = pjoin(data_path, track_name)
        # 如果是真实数据集,将track_name添加到meta_dict中,值为路径.../nocs_fulll/train/00000
        if real and track_name not in meta_dict:
            meta_dict[track_name] = file_path
        # 如果是真实数据集,后缀为depth
        # 如果是合成数据集,后缀为composed
        # 因为合成数据集需要_composed.png来提供深度信息
        suffix = 'depth' if real else 'composed'
        try:
            depth = cv2.imread(pjoin(file_path, f'{prefix}_{suffix}.png'), -1)
            mask = cv2.imread(pjoin(file_path, f'{prefix}_mask.png'))[:, :, 2]
            rgb = cv2.imread(pjoin(file_path, f'{prefix}_color.png'))
            # 读取_meta.txt来获得图中的所有实例
            with open(pjoin(file_path, f'{prefix}_meta.txt'), 'r') as f:
                meta_lines = f.readlines()
            # 读取_pose.pkl来获得图中的所有实例的gt位姿
            # 可以通过pose_dict[i]来获得位姿pose
            with open(pjoin(file_path, f'{prefix}_pose.pkl'), 'rb') as f:
                pose_dict = pickle.load(f)
        except:
            continue
        # 合成数据集flip为True
        # 第二个维度需要颠倒
        if flip:
            depth, mask, rgb = depth[:, ::-1], mask[:, ::-1], rgb[:, ::-1]
        # 初始时inst_num的值为-1
        inst_num = -1
        # 遍历_meta.txt中的每一行, 找到目标instance
        for meta_line in meta_lines:
            # 例:meta_line为"1 6 mug2_scene3_norm"
            # inst_num 为 1
            # inst_num只是一个序号,表明在meta.txt中的第几个,没有其他意义,但是在get_gt_pose中生成的gt位姿是按这个序号来读取的
            # inst_id 为mug2_scene3_norm 与get_instance_list一样
            inst_num = int(meta_line.split()[0])
            inst_id = meta_line.split()[-1]
            # instance是来自instance_list目录下的instance.txt中读取的
            # inst_id则是从_meta.txt中读取的
            # 当meta中的某一行的实例与当前搜索的一致时,跳出循环(看来一张图片中不会出现重复的实例)
            if inst_id == instance:
                break

        # 从pose_dict中读取gt位姿
        if inst_num not in pose_dict:
            continue
        pose = pose_dict[inst_num]
        # bbox的8个点都使用gt位姿进行变换
        posed_bbox = (np.matmul(bbox, pose['rotation'].swapaxes(-1, -2))
                      * np.expand_dims(pose['scale'], (-1, -2))
                      + pose['translation'].swapaxes(-1, -2))
        # 计算位姿变换后包围盒的中心点(相加平均)
        center = posed_bbox.mean(axis=0)
        # 半径为一个顶点到中心点的距离再加0.1
        radius = np.sqrt(np.sum((posed_bbox[0] - center) ** 2)) + 0.1

        # 输入:[中心点减半径,中心点加半径]
        # 输出:np.stack([pmin, pmax] 两个点按大小排列
        aa_corner = get_corners([center - np.ones(3) * radius, center + np.ones(3) * radius])
        # 通过上面的两个点获得新的包围盒aabb
        aabb = bbox_from_corners(aa_corner)

        height, width = mask.shape
        # 将aabb投影到平面,project返回的是像素坐标系的坐标(u,v)
        # [:, [1,0]]  第一个冒号,取全部点;第二个[1,0]取第1和第0列并交换顺序
        projected_corners = project(aabb, intrinsics).astype(np.int32)[:, [1, 0]]
        projected_corners[:, 0] = height - projected_corners[:, 0]
        # 得到2D包围盒的2个顶点
        corner_2d = np.stack([np.min(projected_corners, axis=0),
                              np.max(projected_corners, axis=0)], axis=0)
        # 检查这两个点是否在图像外
        corner_2d[0, :] = np.maximum(corner_2d[0, :], 0)
        corner_2d[1, :] = np.minimum(corner_2d[1, :], np.array([height - 1, width - 1]))
        corner_mask = np.zeros_like(mask)
        # 根据这两个顶点绘制一个矩形(矩形内的值为1),创建corner_mask
        corner_mask[corner_2d[0, 0]: corner_2d[1, 0] + 1, corner_2d[0, 1]: corner_2d[1, 1] + 1] = 1
        # 将_color.png对应的矩形区域裁剪出来
        cropped_rgb = rgb[corner_2d[0, 0]: corner_2d[1, 0] + 1, corner_2d[0, 1]: corner_2d[1, 1] + 1]

        # corner_mask与depth取交集
        # 根据2D bbox反投影得到点云
        raw_pts, raw_idx = backproject(depth, intrinsics=intrinsics, mask=corner_mask)
        # 将2D bbox包含的点和mask包含的点取交集,得到raw_mask
        raw_mask = (mask == inst_num)[raw_idx[0], raw_idx[1]]

        def filter_box(pts, corner):
            mask = np.prod(np.concatenate([pts >= corner[0], pts <= corner[1]], axis=1).astype(np.int8),  # [N, 6]
                           axis=1)
            idx = np.where(mask == 1)[0]
            return pts[idx]

        def filter_ball(pts, center, radius):
            distance = np.sqrt(np.sum((pts - center) ** 2, axis=-1))  # [N]
            idx = np.where(distance <= radius)
            return pts[idx], idx

        # 筛选出raw_pts中再radius为半径的球内的点
        # 这些点再一次对raw_mask进行筛选
        pts, idx = filter_ball(raw_pts, center, radius)
        obj_mask = raw_mask[idx]

        # 将所有信息保存到data_dict中
        # pts是当前实例的点云
        # labels是当前实例点云对应的2D mask, 对应点的值为inst_num,可通过obj_mask[idx]来进行访问
        data_dict = {'points': pts, 'labels': obj_mask, 'pose': pose,
                     'path': pjoin(file_path, f'{prefix}_{suffix}.png')}
        # 合成数据集
        if not real:
            img_num += 1
            # 如果img_num超过限度,则(instance_id)下再创建一个文件夹
            if img_num >= img_per_folder:
                folder_num += 1
                # 例:nocs_data/instance_data/(instance_id)/0001
                cur_folder_path = pjoin(inst_output_path, f'{folder_num:04d}')
                # nocs_data/instance_data/(instance_id)/0001/data
                ensure_dirs([pjoin(cur_folder_path, name) for name in (['data'] if not render_rgb else ['rgb', 'data'])])
                img_num = 0
            # 将data_dict保存至/data/(img_num).npz
            # 例如/data/01.npz
            np.savez_compressed(pjoin(cur_folder_path, 'data', f'{img_num:02d}.npz'), all_dict=data_dict)
            if render_rgb:
                cv2.imwrite(pjoin(cur_folder_path, 'rgb', f'{img_num:02d}.png'), cropped_rgb)
        else:
            cur_folder_path = pjoin(inst_output_path, track_name)
            ensure_dirs(pjoin(cur_folder_path, 'data'))
            np.savez_compressed(pjoin(cur_folder_path, 'data', f'{prefix}.npz'), all_dict=data_dict)

    if real:
        cur_folder_path = pjoin(inst_output_path, track_name)
        ensure_dirs([cur_folder_path])
        for track_name in meta_dict:
            with open(pjoin(cur_folder_path, 'meta.txt'), 'w') as f:
                print(meta_dict[track_name], file=f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../nocs_data')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--list_path', type=str, default='../../nocs_data/instance_list')
    parser.add_argument('--model_path', type=str, default='../../nocs_data/model_corners')
    parser.add_argument('--output_path', type=str, default='../../nocs_data/instance_data')
    parser.add_argument('--category', type=int, default=1)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_proc', type=int, default=10)

    return parser.parse_args()


def main(args):
    if args.data_type in ['real_train', 'real_test']:
        intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    else:
        intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

    data_path = pjoin(args.data_path, args.data_type)  # 数据集路径与数据集类型拼接
    list_path = pjoin(args.list_path, args.data_type, str(args.category))  # get_instance_list.py生成的保存实例路径的目录
    model_path = args.model_path  # 模型所在路径 默认/nocs_data/model_corners
    output_path = pjoin(args.output_path, args.data_type, str(args.category))  # 默认是nocs_data/instance_data
    ensure_dirs(output_path)
    instances = list(map(lambda s: s.split('.')[0], os.listdir(list_path)))  # 去掉.txt 只保留instance这个前缀
    instances.sort()

    if not args.parallel:
        gather_instances(list_path, data_path, model_path, output_path, instances, intrinsics,
                         flip=args.data_type not in ['real_train', 'real_test'],
                         real=args.data_type in ['real_train', 'real_test'])
    else:
        processes = []
        proc_cnt = args.num_proc
        num_per_proc = int((len(instances) - 1) / proc_cnt) + 1

        for k in range(proc_cnt):
            s_ind = num_per_proc * k
            e_ind = min(num_per_proc * (k + 1), len(instances))
            p = Process(target=gather_instances,
                        args=(list_path, data_path, model_path, output_path,
                              instances[s_ind: e_ind], intrinsics,
                              args.data_type not in ['real_train', 'real_test'],
                              args.data_type in ['real_train', 'real_test']))
            processes.append(p)
            p.start()

        """
        for process in processes:
            process.join()
        """


if __name__ == '__main__':
    args = parse_args()
    print('get_instance_data')
    print(args)
    main(args)

