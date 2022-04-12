import os
import numpy as np
import logging  # 引入logging模块
import os.path
import time
import argparse

# 该行是windows下拼接路径,linux也可以用(应该？)
def pjoin(*a):
    path = a[0]
    for i in range(len(a)):
        if i==0:
            continue
        else:
            path = os.path.join(path, a[i]).replace('\\', '/')
    return path


# points_xy:nx2 (x,y) 传入的点是cv提取的2D点,坐标系原点在图像左上角
def backproject_points(points_xy, depth, intrinsics, scale=0.001):
    points_uv1 = np.ones((points_xy.shape[0], 3), np.float64)
    points_uv1[:, 0] = points_xy[:, 0]
    points_uv1[:, 1] = depth.shape[0] - points_xy[:, 1]
    intrinsics_inv = np.linalg.inv(intrinsics)

    points_xyz = intrinsics_inv @ points_uv1.transpose()
    points_xyz = np.transpose(points_xyz)
    for i in range(points_xy.shape[0]):
        kp = points_xy[i, :]
        kp_x = kp[0]
        kp_y = kp[1]
        point_depth = depth[kp_y, kp_x].astype(np.float32)
        points_xyz[i, :] = points_xyz[i, :] * point_depth / points_xyz[i, -1:]
    points_xyz[:, 2] = -points_xyz[:, 2]
    zero_idx = np.where(points_xyz[:, 2] == 0)
    return points_xyz*scale, zero_idx


def add_border(mask, inst_id, kernel_size=10):  # enlarge the region w/ 255
    # print((255 - mask).sum())
    output = mask.copy()
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i][j] == inst_id:
                output[max(0, i - kernel_size): min(h, i + kernel_size),
                max(0, j - kernel_size): min(w, j + kernel_size)] = inst_id
    # print((255 - output).sum())
    return output


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=None, help='')
    parser.add_argument('--dataset', type=str, default=None, help='')
    parser.add_argument('--mode', type=str, default=None, help='')
    parser.add_argument('--obj_model', type=str, default='M:/PCL/dataset/obj_models_CAPTRA/', help='')

    opt = parser.parse_args()
    return opt


def init_logger(name):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = pjoin(os.getcwd(), 'log')
    log_name = pjoin(log_path, rq + '_' + name +'.log')
    logfile = log_name
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    return logger