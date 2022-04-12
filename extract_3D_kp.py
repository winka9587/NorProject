import os
from extract_2D_kp import extract_sift_kp_from_RGB
from main import read_scene_imgs, add_border
from utils import pjoin, config, viz_multi_points_diff_color, RotMatErr, init_logger
import cv2
import sys
import time
import numpy as np
from main import backproject_points
from captra_utils.pose import get_instance_pose
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
        self.meta_idx = meta_idx  # ����ͨ�����indexȥmeta_s��ֱ�ӷ��ʵ�ǰģ�Ͷ�Ӧ��meta��Ϣ
        self.meta_s = meta_s
        self.inst_id = inst_id
        self.crop_xmin = crop_xmin  # ���ڻ�òü����ͼ������uv��ԭͼ�ϵ�����uv,�������ڲ���˷�ͶӰ
        self.crop_ymin = crop_ymin


def extract_sift_feature(arg):
    root_path = arg.root_path
    dataset = arg.dataset
    mode = arg.mode
    inst = arg.inst
    prefix = arg.prefix
    obj_name = arg.obj_name
    opt = arg.opt
    # ��ȡ3D��
    color, depth, coord, mask, meta_s = read_scene_imgs(root_path, dataset, mode, inst, prefix, opt)
    instance_id = -1
    meta_idx = 0
    for meta in meta_s:
        if meta["model_name"] == obj_name:
            instance_id = meta["inst_idx"]
            break
        else:
            meta_idx += 1
    # ��ȡobj��Ӧ��mask
    mask_obj = np.zeros((depth.shape[0], depth.shape[1], 3), np.float64)
    mask_obj.fill(255)
    mask_obj[np.where(mask == instance_id)] = 0
    # if arg.vis:
    #     cv2.imshow("obj mask origin", mask_obj)
    #     cv2.waitKey(0)
    # addborder����mask_add
    mask_add = add_border(mask, instance_id, kernel_size=10)
    mask_obj = np.zeros((depth.shape[0], depth.shape[1], 3), np.float64)
    mask_obj.fill(255)
    mask_obj[np.where(mask_add == instance_id)] = 0
    # if arg.vis:
    #     cv2.imshow("obj mask addborder", mask_obj)
    #     cv2.waitKey(0)
    # mask��ȡ2D bbox,�и�rgbͼ��
    idx = np.where(mask_add == instance_id)
    x_max = max(idx[1])
    x_min = min(idx[1])
    y_max = max(idx[0])
    y_min = min(idx[0])
    print(f'x:[{x_min}:{x_max}], y[{y_min}:{y_max}]')
    color_cropped = color[y_min:y_max, x_min:x_max, :]
    depth_cropped = depth[y_min:y_max+1, x_min:x_max+1]  # depthͼ���ǵ�ͨ��, +1����Ϊ��ȡ���߽��ϵ�ֵ��+1����Ӱ���������㣬��ֹԽ��
    # if arg.vis:
    #     cv2.imshow("color cut", color_cropped)
    #     cv2.waitKey(0)
    # ��ȡ����rgbͼ����ȡsift������
    # color_sift ����sift�������RGBͼ��
    color_sift, kp_xys, des = extract_sift_kp_from_RGB(opt, color_cropped)
    if arg.vis:
        cv2.imshow("color cut sift", color_sift)
        cv2.waitKey(0)
# ֱ�ӻ��obj��Ӧ��meta��һ�У�������������������������������������������������������������������������������������������������������������������������
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
        # ���ӻ�siftƥ����
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

    # # ������ĵ㷴ͶӰ,�������ʹ��Umeyama�㷨����RT
    # # # ���Է�ͶӰ,�õ�2D���Ӧ��3D���� point_3d: nx3
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
    pr