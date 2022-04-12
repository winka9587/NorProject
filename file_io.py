import numpy as np
import cv2
from utils import pjoin


# 给定数据集(CAMERA/Real),mode(train/val/test),实例inst,图片前缀prefix
# 来获取_color,_depth,_mask,_meta 4个文件并进行处理
# 返回处理过的depth, coord, mask, lines(meta文件中的每一行)
def read_scene_imgs(root_path, dataset, mode, inst, prefix, obj_name, opt):
    global pcd_save_path
    global kernel_size
    if dataset == 'CAMERA':
        flip = True
    else:
        flip = False

    file_path = pjoin(root_path, dataset, mode, inst)
    color = cv2.imread(pjoin(file_path, f'{prefix}_color.png'))
    depth = cv2.imread(pjoin(file_path, f'{prefix}_depth.png'), -1)
    coord = cv2.imread(pjoin(file_path, f'{prefix}_coord.png'))
    mask = cv2.imread(pjoin(file_path, f'{prefix}_mask.png'))
    with open(pjoin(file_path, f'{prefix}_meta.txt'), 'r') as f:
        lines = f.readlines()

    meta_s = []
    for line in lines:
        elements = line.strip().split(' ')
        meta = {}
        meta['inst_idx'] = elements[0]
        meta['cat_id'] = elements[1]
        if dataset == 'Real':
            meta['parent_dir'] = None
            meta['obj_name'] = elements[2]
        if dataset == 'CAMERA':
            meta['parent_dir'] = elements[2]
            meta['obj_name'] = elements[3]
        meta_s.append(meta)

    if depth is None or coord is None or mask is None:
        print(pjoin(file_path, f'{prefix}_xxx.png is None'))
        return None, None, None, None
    if len(depth.shape) == 3:
        depth = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])
        depth = depth.astype(np.uint16)
    mask = mask[:, :, 2]

    # 水平翻转
    if flip:
        depth, coord, mask = depth[:, ::-1], coord[:, ::-1], mask[:, ::-1]

    # plot_images([coord, mask])
    # 相当于调换了NOCS map中xyz的值
    # 并使其范围在[-0.5, 0.5]
    coord = coord[:, :, (2, 1, 0)]
    coord = coord / 255. - 0.5
    if not flip:
        coord[..., 2] = -coord[..., 2]  # verify!!!

    return color, depth, coord, mask, meta_s