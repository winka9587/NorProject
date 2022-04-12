from extract_2D_kp import extract_sift_kp_from_RGB
from main import read_scene_imgs, add_border
from utils import config
import cv2
import sys
import time
import numpy as np


class InputArg:
    def __init__(self, root_path, dataset, mode, inst, prefix, obj_name, opt, vis=True):
        self.root_path = root_path
        self.dataset = dataset  # Real or CAMERA
        self.mode = mode
        self.inst = inst
        self.prefix = prefix
        self.obj_name = obj_name
        self.opt = opt
        self.vis = vis


class OutputArg:
    def __init__(self, kp, des, color_sift):
        self.kp = kp
        self.des = des
        self.color_sift = color_sift


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
    for meta in meta_s:
        if meta["model_name"] == obj_name:
            instance_id = meta["inst_idx"]
            break
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
    color_mask = color[y_min:y_max, x_min:x_max, :]
    # if arg.vis:
    #     cv2.imshow("color cut", color_mask)
    #     cv2.waitKey(0)
    # 提取出的rgb图像提取sift特征点
    color_sift, kp_xys, des = extract_sift_kp_from_RGB(opt, color_mask)
    if arg.vis:
        cv2.imshow("color cut sift", color_sift)
        cv2.waitKey(0)

    output = OutputArg(kp_xys, des, color_sift)
    return output

if __name__ == '__main__':
    # use opencv to detect feature and descriptor, then match
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    opt = config()
    root_path = 'H:/PCL/'
    dataset = 'Real'  # Real or CAMERA
    mode = 'train'
    inst = 'scene_4'
    prefix = '0000'
    obj_name = 'laptop_air_0_norm'

    arg_1 = InputArg(root_path, dataset, mode, inst, '0000', obj_name, opt)
    output_1 = extract_sift_feature(arg_1)

    arg_2 = InputArg(root_path, dataset, mode, inst, '0001', obj_name, opt)
    output_2 = extract_sift_feature(arg_2)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(output_1.des, output_2.des, k=2)


    def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
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

        cv2.namedWindow("match", cv2.WINDOW_NORMAL)
        cv2.imshow("match", vis)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50 * n.distance:
            goodMatch.append(m)

    # drawMatchesKnn_cv2(output_1.color_sift, output_1.kp, output_2.color_sift, output_2.kp, goodMatch[:20])
    drawMatchesKnn_cv2(output_1.color_sift, output_1.kp, output_2.color_sift, output_2.kp, goodMatch[:])

    cv2.waitKey(0)
    cv2.destroyAllWindows()