import os
from extract_2D_kp import extract_sift_kp_from_RGB
from main import read_scene_imgs, add_border
from utils import pjoin, config, viz_multi_points_diff_color, RotMatErr, init_logger
import cv2
import sys
import time
import numpy as np
from main import backproject_points
from captra_utils.pose import get_instance_pose, pose2sRt, sRT2pose
from captra_utils.align_pose import pose_fit
from captra_utils.utils_from_captra import backproject

logger = init_logger('AlignPcdOnlyBy2DFeature')

class InputArg:
    def __init__(self, root_path, dataset, mode, inst, prefix, obj_name, opt, vis=False):
        self.root_path = root_path
        self.dataset = dataset  # Real or CAMERA
        self.mode = mode
        self.inst = inst
        self.prefix = prefix
        self.obj_name = obj_name
        self.opt = opt
        self.vis = vis


class OutputArg:
    def __init__(self, kp, des, color_sift, color_cropped, depth_cropped, color, depth, coord, mask, meta_idx, meta_s, inst_id, crop_xmin, crop_ymin):
        self.kp = kp
        self.des = des
        self.color_sift = color_sift
        self.color_cropped = color_cropped
        self.depth_cropped = depth_cropped
        self.color = color
        self.depth = depth
        self.coord = coord
        self.mask = mask
        self.meta_idx = meta_idx  # 可以通过这个index去meta_s中直接访问当前模型对应的meta信息
        self.meta_s = meta_s
        self.inst_id = inst_id
        self.crop_xmin = crop_xmin  # 用于获得裁剪后的图像坐标uv在原图上的坐标uv,用于与内参相乘反投影
        self.crop_ymin = crop_ymin


def extract_sift_feature(arg):
    root_path = arg.root_path
    dataset = arg.dataset
    mode = arg.mode
    inst = arg.inst
    prefix = arg.prefix
    obj_name = arg.obj_name
    opt = arg.opt
    # 提取3D点
    color, depth, coord, mask, meta_s = read_scene_imgs(root_path, dataset, mode, inst, prefix, opt)
    instance_id = -1
    meta_idx = 0
    for meta in meta_s:
        if meta["model_name"] == obj_name:
            instance_id = meta["inst_idx"]
            break
        else:
            meta_idx += 1
    # 提取obj对应的mask
    mask_obj = np.zeros((depth.shape[0], depth.shape[1], 3), np.float64)
    mask_obj.fill(255)
    mask_obj[np.where(mask == instance_id)] = 0
    # if arg.vis:
    #     cv2.imshow("obj mask origin", mask_obj)
    #     cv2.waitKey(0)
    # addborder生成mask_add
    mask_add = add_border(mask, instance_id, kernel_size=10)
    mask_obj = np.zeros((depth.shape[0], depth.shape[1], 3), np.float64)
    mask_obj.fill(255)
    mask_obj[np.where(mask_add == instance_id)] = 0
    # if arg.vis:
    #     cv2.imshow("obj mask addborder", mask_obj)
    #     cv2.waitKey(0)
    # mask获取2D bbox,切割rgb图像
    idx = np.where(mask_add == instance_id)
    x_max = max(idx[1])
    x_min = min(idx[1])
    y_max = max(idx[0])
    y_min = min(idx[0])
    print(f'x:[{x_min}:{x_max}], y[{y_min}:{y_max}]')
    color_cropped = color[y_min:y_max, x_min:x_max, :]
    depth_cropped = depth[y_min:y_max+1, x_min:x_max+1]  # depth图像是单通道, +1是因为会取到边界上的值，+1不会影响坐标的起点，防止越界
    # if arg.vis:
    #     cv2.imshow("color cut", color_cropped)
    #     cv2.waitKey(0)
    # 提取出的rgb图像提取sift特征点
    # color_sift 绘制sift特征点的RGB图像
    color_sift, kp_xys, des = extract_sift_kp_from_RGB(opt, color_cropped)
    if arg.vis:
        cv2.imshow("color cut sift", color_sift)
        cv2.waitKey(0)
# 直接获得obj对应的meta那一行！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    output = OutputArg(kp=kp_xys,
                       des=des,
                       color_sift=color_sift,
                       color_cropped=color_cropped,
                       depth_cropped=depth_cropped,
                       color=color,
                       depth=depth,
                       coord=coord,
                       mask=mask,
                       meta_idx=meta_idx,
                       meta_s=meta_s,
                       inst_id=instance_id,
                       crop_xmin = x_min,
                       crop_ymin = y_min
                       )
    return output

def get_sift_error_between_two_frame(prefix_1, prefix_2, opt=None):
    # use opencv to detect feature and descriptor, then match
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    arg_1 = InputArg(root_path, dataset, mode, inst, prefix_1, obj_name, opt)
    output_1 = extract_sift_feature(arg_1)

    arg_2 = InputArg(root_path, dataset, mode, inst, prefix_2, obj_name, opt)
    output_2 = extract_sift_feature(arg_2)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(output_1.des, output_2.des, k=2)


    def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
        points_1 = []
        points_2 = []
        h1, w1 = img1_gray.shape[:2]
        h2, w2 = img2_gray.shape[:2]

        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = img1_gray
        vis[:h2, w1:w1 + w2] = img2_gray

        p1 = [kpp.queryIdx for kpp in goodMatch]
        p2 = [kpp.trainIdx for kpp in goodMatch]

        post1 = np.int32([kp1[pp] for pp in p1])
        post2 = np.int32([kp2[pp] for pp in p2]) + (w1, 0)

        for (x1, y1), (x2, y2) in zip(post1, post2):
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            points_1.append([x1, y1])
            points_2.append([x2, y2])
        # 可视化sift匹配结果
        # cv2.namedWindow("match", cv2.WINDOW_NORMAL)
        # cv2.imshow("match", vis)

        match_points_1 = np.asarray(points_1)
        match_points_2 = np.asarray(points_2)
        return match_points_1, match_points_2-(w1, 0)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50 * n.distance:
            goodMatch.append(m)

    # drawMatchesKnn_cv2(output_1.color_sift, output_1.kp, output_2.color_sift, output_2.kp, goodMatch[:20])
    match1, match2 = drawMatchesKnn_cv2(output_1.color_sift, output_1.kp, output_2.color_sift, output_2.kp, goodMatch[:])
    print(f'match1:{match1.shape}')
    print(f'match2:{match2.shape}')

    # # 将计算的点反投影,两组点云使用Umeyama算法计算RT
    # # # 测试反投影,得到2D点对应的3D点云 point_3d: nx3
    real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    point_3d_1, depth_zero_idx_1, point_3d_1_origin = backproject_points(match1,
                                    output_1.depth_cropped,
                                    intrinsics=real_intrinsics,
                                    xmin=output_1.crop_xmin,
                                    ymin=output_1.crop_ymin,
                                    img=output_1.color)
    point_3d_2, depth_zero_idx_2, point_3d_2_origin = backproject_points(match2,
                                    output_2.depth_cropped,
                                    intrinsics=real_intrinsics,
                                    xmin=output_2.crop_xmin,
                                    ymin=output_2.crop_ymin,
                                    img=output_2.color)
    print(f'match1 project origin:{point_3d_1_origin.shape}')
    print(f'match2 project origin:{point_3d_2_origin.shape}')
    detph_zero_idx = np.concatenate((depth_zero_idx_1, depth_zero_idx_2), axis=1)
    point_3d_1 = np.delete(point_3d_1_origin, detph_zero_idx, axis=0)
    point_3d_2 = np.delete(point_3d_2_origin, detph_zero_idx, axis=0)
    print(f'match1 project non zero:{point_3d_1.shape}')
    print(f'match2 project non zero:{point_3d_2.shape}')

    print('point')

    # mask_kp_11 = np.zeros((output_1.depth_cropped.shape[0], output_1.depth_cropped.shape[1], 3), np.float64)
    # mask_kp_11.fill(255)
    # mask_kp_11[match1[:, 1], match1[:, 0], :] = 0
    # cv2.imshow('1', mask_kp_11)
    # cv2.waitKey(0)

    color_purple = np.array([150, 0, 255])
    color_red = np.array([255, 0, 0])
    color_green = np.array([0, 255, 0])
    color_blue = np.array([0, 0, 255])
    # viz
    # viz_multi_points_diff_color("3D kp from two frame", [point_3d_1, point_3d_2], [color_green, color_blue], False, True)

    pose1 = get_instance_pose(output_1.meta_s[output_1.meta_idx], output_1.mask, output_1.coord, output_1.depth, real_intrinsics,
                           output_1.inst_id, opt)

    pose2 = get_instance_pose(output_2.meta_s[output_2.meta_idx], output_2.mask, output_2.coord, output_2.depth, real_intrinsics,
                           output_2.inst_id, opt)
    print(f'point1:{point_3d_1.shape}')
    print(f'point2:{point_3d_2.shape}')
    point_3d_1[:, :] = point_3d_1[:, :] * 0.001
    point_3d_2[:, :] = point_3d_2[:, :] * 0.001
    pose_sift = pose_fit(point_3d_1, point_3d_2)

    s12 = pose2['scale']/pose1['scale']
    t12 = pose2['translation'] - pose1['translation']
    R12 = np.matmul(pose2['rotation'], pose1['rotation'].transpose())
    err_t_tmp = pose_sift['translation']-t12

    err_t = np.sqrt(np.square(err_t_tmp[0])+np.square(err_t_tmp[1])+np.square(err_t_tmp[2]))
    err_R_degree = RotMatErr(R12, pose_sift['rotation'])
    err_s = abs(pose_sift['scale']/(s12))

    print("pose two frame (sift):")
    print(pose_sift)
    pose_diff_gt = {}
    pose_diff_gt['rotation'] = R12
    pose_diff_gt['scale'] = s12
    pose_diff_gt['translation'] = t12
    print("pose_diff_gt:")
    print(pose_diff_gt)
    print("pose1:")
    print(pose1)
    print("pose2:")
    print(pose2)

    print(f'err_s: {err_s}')
    print(f'err_R: {err_R_degree}')
    print(f'err_t: {err_t}')

    point_3d_2_sift = (np.matmul(pose_sift['rotation'], point_3d_1.transpose()).transpose()) * pose_sift['scale'] + pose_sift['translation'].transpose()
    point_3d_2_gt = (np.matmul(pose_diff_gt['rotation'], point_3d_1.transpose()).transpose()) * pose_diff_gt['scale'] + pose_diff_gt['translation'].transpose()

    # viz_multi_points_diff_color("3D kp from two frame",
    #                             [point_3d_2, point_3d_2_gt, point_3d_2_sift],
    #                             [color_blue, color_red, color_purple],
    #                             False,
    #                             True)
    print('point_3d_1')
    print('point_3d_2')
    # viz_multi_points_diff_color("green:Frame1 kp, blue:Frame2 kp, red:Frame1 kp use gt to Frame2",
    #                             [point_3d_1, point_3d_2, point_3d_2_gt, point_3d_2_sift],
    #                             [color_green, color_blue, color_red, color_purple],
    #                             False,
    #                             True)
    result_err = {}
    result_err['s'] = err_s
    result_err['R'] = err_R_degree
    result_err['t'] = err_t
    print('end')
    return result_err

if __name__ == '__main__':
    opt = config()
    root_path = 'M:/PCL/'
    dataset = 'Real'  # Real or CAMERA
    mode = 'train'
    inst = 'scene_4'
    prefix = '0000'
    obj_name = 'laptop_air_0_norm'
    opt.root_path = root_path
    opt.dataset = dataset
    opt.mode = mode
    file_path = pjoin(root_path, 'dataset', dataset, mode, inst)
    files = os.listdir(file_path)
    prefixs = [filename[:-10] for filename in files if filename[-10:]=='_color.png']
    avg_err = {}
    avg_err['s'] = 0
    avg_err['R'] = 0
    avg_err['t'] = 0
    avg_err['count'] = 0
    print(f"prefixs = :\n{prefixs}")
    # for i in range(len(prefixs)-1):
    #     prefix_1 = prefixs[i]
    #     prefix_2 = prefixs[i+1]
    #     result_err = get_sift_error_between_two_frame(prefix_1, prefix_2, opt)
    #     avg_err['s'] += result_err['s']
    #     avg_err['R'] += result_err['R']
    #     avg_err['t'] += result_err['t']
    #     avg_err['count'] += 1
    #
    #     logger.info(f'match img {prefix_1}, {prefix_2}')
    #     logger.info('Error')
    #     logger.info('s:{}'.format(result_err['s']))
    #     logger.info('R:{}'.format(result_err['R']))
    #     logger.info('t:{}'.format(result_err['t']))

    result_err = get_sift_error_between_two_frame('0000', '0001', opt)


