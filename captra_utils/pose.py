from align_pose import pose_fit
from utils import pjoin
from utils_from_captra import sample_points_from_mesh, backproject
import numpy as np


def get_instance_pose(meta, mask, coord, depth, intrinsics, inst_i, opt):
    # inst_model_name = inst_meta_line.split(' ')[3]  # 合成数据集
    pose_dict = {}
    # 从pkl加载物体gt模型
    # CAMERA数据集
    # pkl_name = ''
    # if opt["dataset"] == 'Real':
    #     pkl_name += "Real_"
    # else:
    #     pkl_name += "camera_"
    # if opt["mode"] == 'train':
    #     pkl_name += 'train'
    # else:
    #     pkl_name += 'test'
    # obj_model_path = pjoin(opt["obj_model_path"], pkl_name) + '.pkl'
    # gt_model_pkl = open(obj_model_path, 'rb')
    # print(f'load {inst_model_name}')
    # gt_models = cPickle.load(gt_model_pkl)
    # gt_model_numpy = gt_models[inst_model_name]

    # 直接加载原obj模型
    if opt.dataset == "CAMERA":
        obj_model_path = pjoin(opt.obj_model_path, opt.dataset, opt.mode, meta["model_parent_dir"],
                               meta["model_name"], 'model.obj')
    else:
        obj_model_path = pjoin(opt.obj_model_path, opt.dataset, opt.mode,
                               meta["model_name"] + '.obj')
    gt_model_numpy = sample_points_from_mesh(obj_model_path, 2048, fps=True, ratio=3)
    # flip z-axis in CAMERA   # 反转Z轴
    # model_points = model_points * np.array([[1.0, 1.0, -1.0]])
    # gt_models = cPickle.load(obj_model_path)
    # gt_model_numpy = gt_models[inst_model_name]

    # 加载prior
    # catId = 1
    # names_and_pcds = np.load(pjoin(opt.root_path, 'mean_shape/prior.npy'), allow_pickle=True)
    # names = names_and_pcds[0]
    # mean_shapes = names_and_pcds[1]
    # prior = mean_shapes[catId]


    # for i in range(1, num_instances + 1):
    #     # 某一个instance在mask中占的像素值少于3个说明没有得到该物体
    #     if np.sum(mask == i) < 3:
    #         continue
    #     # 将深度图反投影到3D平面
    #     # pts为3D点，idxs为对应的2D坐标，可以使用它去coord中提取对应的点
    #     pts, idxs = backproject(depth, intrinsics, mask == i)
    #     coord_pts = coord[idxs[0], idxs[1], :]  # already centered
    #     if len(pts) < 3:
    #         continue
    #     # plot3d_pts([[pts], [coord_pts]])
    #     # 深度得到的3D坐标和NOCS的3D坐标进行位姿拟合
    #     pose = pose_fit(coord_pts, pts)
    #     if pose is not None:
    #         pose_dict[i] = pose

    # 某一个instance在mask中占的像素值少于3个说明没有得到该物体
    if np.sum(mask == inst_i) < 3:
        print(f'pixel of instance {inst_i} not enough')
    # 将深度图反投影到3D平面
    # pts为3D点，idxs为对应的2D坐标，可以使用它去coord中提取对应的点
    pts, idxs = backproject(depth, intrinsics, mask == inst_i)
    # 得到对应的2D和3D点坐标，使用2D坐标去nocs图中读取对应的nocs坐标，就有了3D坐标(相机坐标系)和3D坐标(nocs)的对应
    coord_pts = coord[idxs[0], idxs[1], :]  # already centered
    if len(pts) < 3:
        print(f'3D points of instance {inst_i} not enough')
    # plot3d_pts([[pts], [coord_pts]])
    # 深度得到的3D坐标(观测点云)和NOCS的3D坐标进行位姿拟合
    print(f'pose fit points num {len(pts)}')
    pose = pose_fit(coord_pts, pts)
    return pose


def pose2sRt(pose):
    sRt = np.zeros((4, 4), np.float64)
    sRt[:3, :3] = pose['rotation']
    sRt[:3, 3] = pose['translation'].transpose()
    sRt[3, 3] = 1
    return sRt


def sRT2pose(sRt):
    pose = {}
    pose['translation'] = sRt[:3, 3].transpose()
    pose['rotation'] = sRt[:3, :3]
    pose['scale'] = 1.0
    return pose
