import os
import numpy as np


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
        x = kp[0]
        y = kp[1]
        point_depth = depth[y, x].astype(np.float32)
        points_xyz[i, :] = points_xyz[i, :] * point_depth/point_depth[:, -1:]
    points_xyz[:, 2] = -points_xyz[:, 2]
    zero_idx = np.where(points_xyz[:, 2] == 0)
    return points_xyz*scale, zero_idx
