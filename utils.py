import os
import numpy as np
import logging  # 引入logging模块
import os.path
import time
import argparse
import torch
# 该行是windows下拼接路径,linux也可以用(应该？)
def pjoin(*a):
    path = a[0]
    for i in range(len(a)):
        if i==0:
            continue
        else:
            path = os.path.join(path, a[i]).replace('\\', '/')
    return path

# mine
# 将离散的2D点反投影到3D空间中
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
    zero_idx = np.where(points_xyz[:, 2] == 0)  # only for debug
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


def add_border_bool(mask, kernel_size=10):  # enlarge the region w/ 255
    # print((255 - mask).sum())
    if isinstance(mask, torch.Tensor):
        output = mask.clone()
    else:
        output = mask.copy()
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i][j] == True:
                output[max(0, i - kernel_size): min(h, i + kernel_size),
                max(0, j - kernel_size): min(w, j + kernel_size)] = True
    # print((255 - output).sum())
    return output


def add_border_bool_by_crop_pos(mask, crop_pos, kernel_size=10):  # enlarge the region w/ 255
    # print((255 - mask).sum())
    if isinstance(mask, torch.Tensor):
        output = mask.clone()
    else:
        output = mask.copy()
    h, w = mask.shape
    output[max(0, crop_pos['rmin'] - kernel_size): min(h, crop_pos['rmax'] + kernel_size),
            max(0, crop_pos['cmin'] - kernel_size): min(w, crop_pos['cmax'] + kernel_size)] = True
    # print((255 - output).sum())
    return output


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=None, help='')
    parser.add_argument('--dataset', type=str, default=None, help='')
    parser.add_argument('--mode', type=str, default=None, help='')
    parser.add_argument('--obj_model', type=str, default='M:/PCL/dataset/obj_models_CAPTRA/', help='')
    parser.add_argument('--vis', type=bool, default=False, help='')
    opt = parser.parse_args()
    return opt


def init_logger(name):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
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
    logger.info('program start')
    return logger


def ensure_dir(path, verbose=False):
    print("ensure_dir: ")
    if not os.path.exists(path):
        if verbose:
            print("     Create folder ", path)
        os.makedirs(path)
    else:
        if verbose:
            print(f"    {path} already exists.")


def ensure_dirs(paths):
    if isinstance(paths, list):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


class Timer:
    def __init__(self, on):
        self.on = on
        self.cur = time.time()

    def tick(self, str=None):
        if not self.on:
            return
        cur = time.time()
        diff = cur - self.cur
        self.cur = cur
        if str is not None:
            print(str, diff)
        return diff


def cvt_torch(x, device):
    if isinstance(x, np.ndarray):
        return torch.tensor(x).float().to(device)
    elif isinstance(x, torch.Tensor):
        return x.float().to(device)
    elif isinstance(x, dict):
        return {key: cvt_torch(value, device) for key, value in x.items()}
    elif isinstance(x, list):
        return [cvt_torch(item, device) for item in x]
    elif x is None:
        return None


# get square image crop window pos
def get_bbox(bbox):
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

