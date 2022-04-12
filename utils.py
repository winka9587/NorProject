import os
import numpy as np
import logging  # ����loggingģ��
import os.path
import time
import argparse

# ������windows��ƴ��·��,linuxҲ������(Ӧ�ã�)
def pjoin(*a):
    path = a[0]
    for i in range(len(a)):
        if i==0:
            continue
        else:
            path = os.path.join(path, a[i]).replace('\\', '/')
    return path


# points_xy:nx2 (x,y) ����ĵ���cv��ȡ��2D��,����ϵԭ����ͼ�����Ͻ�
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
    # ��һ��������һ��logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log�ȼ��ܿ���
    # �ڶ���������һ��handler������д����־�ļ�
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = pjoin(os.getcwd(), 'log')
    log_name = pjoin(log_path, rq + '_' + name +'.log')
    logfile = log_name
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # �����file��log�ȼ��Ŀ���
    # ������������handler�������ʽ
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # ���Ĳ�����logger��ӵ�handler����
    logger.addHandler(fh)
    return logger