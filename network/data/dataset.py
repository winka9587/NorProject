import math
import numpy as np
import torch.utils.data as data
import os
from os.path import join as pjoin
import cv2
from utils import ensure_dir
from captra_utils.utils_from_captra import load_depth
import _pickle as cPickle
from dataset_process import split_nocs_dataset


# result_path:
#       --img_list
# mode: [train, test]
# source: [Real, CAMERA, Real_CAMERA]
class NOCSDataset(data.Dataset):
    def __init__(self, dataset_path, result_path, obj_category, mode, n_points=4096, bad_ins=None, opt=None):
        assert (mode == 'train' or mode == 'val'), 'Mode must be "train" or "val".'
        self.dataset_path = dataset_path
        self.result_path = result_path
        self.obj_category = obj_category
        self.mode = mode
        self.n_points = n_points
        self.opt = opt
        self.file_list = self.collect_data()
        self.len = len(self.file_list)
        self.bad_ins = bad_ins  # bad_ins��һ������,�洢ʵ��ģ�����������õ�ʵ�������ݼ�file_list���޳�


    def collect_data(self):
        # ../data/nocs_data/splits/1/1_bottle_rot
        splits_path = pjoin(self.dataset_path, "splits", self.obj_category, self.num_expr)
        idx_txt = pjoin(splits_path, f'{self.mode}.txt')
        print(f'NOCSDataSet load data from {idx_txt}')
        # ���û��{mode}.txt��ʹ��split_nocs_dataset��������һ��
        splits_ready = os.path.exists(idx_txt)
        if not splits_ready:
            split_nocs_dataset(self.root_dset, self.obj_category, self.num_expr, self.mode, self.bad_ins)
        # ��ȡmode.txt�ļ�
        with open(idx_txt, "r", errors='replace') as fp:
            lines = fp.readlines()
            file_list = [line.strip() for line in lines]  # ��ÿһ�е�npz�ļ�·������file_list��
        # if downsampling is not None:
        #     file_list = file_list[::downsampling]
        #
        # if truncate_length is not None:
        #     file_list = file_list[:truncate_length]
        return file_list

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # ����index��ƴ��img_list�е�·��
        img_path = pjoin(self.dataset_path, self.img_list[index])
        gt_label_path = pjoin(self.result_path, 'gt_label', self.img_list[index])
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]  # ��ȡ��ͨ����ɫͼ��
        rgb = rgb[:, :, ::-1]  # ���һ��ͨ������
        depth = load_depth(img_path)  # ���ͼֻ�е�ͨ��
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2]  # maskֻ��Ҫһ��ͨ������
        coord = cv2.imread(img_path + '_coord.png')[:, :, :3]  # coord��Ҫǰ����ͨ��
        coord = coord[:, :, (2, 1, 0)]  # �ߵ���ͨ��˳��
        coord = np.array(coord, dtype=np.float32) / 255  # nocsͼֵΪ0��1
        coord[:, :, 2] = 1 - coord[:, :, 2]
        # ��ȡpkl�ļ������ground truth
        # ���д洢�����ǩclass_ids,��Χ��bboxes(2D), �ߴ�scale, rotations, translations, ʵ��instance_ids, model_list
        with open(gt_label_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        # ��ȡ����ڲ�
        #if 'CAMERA' in img_path.split('/'):
        if 'CAMERA' == self.img_list[index].split('/')[0]:
            cam_fx, cam_fy, cam_cx, cam_cy = self.camera_intrinsics
        else:
            cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics

        # select one foreground object
        # ���ѡ��ͼƬ��һ��ʵ��
        idx = random.randint(0, len(gts['instance_ids']) - 1)
        inst_id = gts['instance_ids'][idx]
        # ��ȡbbox
        # ����4������
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        # sample points
        # ���ݻ�ø�ʵ����Ӧ��mask,��ʹ��mask��depthһ���õ�
        mask = np.equal(mask, inst_id)
        mask = np.logical_and(mask, depth > 0)
        # mask������ȡֵ����inst_id������
        # Ȼ���depth�����߼�and�����õ�����
        # ƽ�̺�õ�����Ԫ��λ��
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) > self.n_pts:
            # �����̫�࣬���������Ҫ����n_pts�ĵ�
            # ����һ����chooseһ������ȫ��һά����,Ȼ��ǰn_pts���㸳ֵΪ1
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.n_pts] = 1
            np.random.shuffle(c_mask)
            # �������
            choose = choose[c_mask.nonzero()]
        else:
            # ����������С�ڵ���n_pts,��乻n_pts
            choose = np.pad(choose, (0, self.n_pts - len(choose)), 'wrap')
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]  # (n_pts,1)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        pt2 = depth_masked / self.norm_scale
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        points = np.concatenate((pt0, pt1, pt2), axis=1)  # ��Ȼָ�����
        # reshape((-1,3))��(H,W,3)->(H*W,3) ʹ��choose������Ȼ�ܹ���ö�Ӧ��ֵ
        # ��nocsͼ��ȡ����Ӧ�ĵ�����,��ȥ0.5
        nocs = coord[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :] - 0.5
        # resize cropped image to standard size and adjust 'choose' accordingly
        rgb = rgb[rmin:rmax, cmin:cmax, :]
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)
        # label
        cat_id = gts['class_ids'][idx] - 1  # convert to 0-indexed
        model = self.models[gts['model_list'][idx]].astype(np.float32)  # 1024 points
        prior = self.mean_shapes[cat_id].astype(np.float32)
        scale = gts['scales'][idx]
        rotation = gts['rotations'][idx]
        translation = gts['translations'][idx]
        # data augmentation
        # ѵ��ģʽ��,�Բ�ɫͼrgb,����points��ʩ�Ӹ���
        if self.mode == 'train':
            # color jitter
            rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
            rgb = np.array(rgb)
            # point shift
            add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
            translation = translation + add_t[0]
            # point jitter
            add_t = add_t + np.clip(0.001 * np.random.randn(points.shape[0], 3), -0.005, 0.005)
            points = np.add(points, add_t)
        rgb = self.transform(rgb)
        points = points.astype(np.float32)
        # adjust nocs coords for mug category
        if cat_id == 5:
            T0 = self.mug_meta[gts['model_list'][idx]][0]
            s0 = self.mug_meta[gts['model_list'][idx]][1]
            nocs = s0 * (nocs + T0)
        # map ambiguous rotation to canonical rotation
        if cat_id in self.sym_ids:
            rotation = gts['rotations'][idx]
            # assume continuous axis rotation symmetry
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x ** 2 + theta_y ** 2)
            s_map = np.array([[theta_x / r_norm, 0.0, -theta_y / r_norm],
                              [0.0, 1.0, 0.0],
                              [theta_y / r_norm, 0.0, theta_x / r_norm]])
            rotation = rotation @ s_map
            nocs = nocs @ s_map
        sRT = np.identity(4, dtype=np.float32)
        sRT[:3, :3] = scale * rotation
        sRT[:3, 3] = translation
        nocs = nocs.astype(np.float32)

        return points, rgb, choose, cat_id, model, prior, sRT, nocs

if __name__ == "__main__":
    dataset_path = '/data1/cxx/Lab_work/dataset'
    result_path = '/data1/cxx/Lab_work/results'
    obj_category = 1
    mode = 'train'
    dataset = NOCSDataset(dataset_path,
                          result_path,
                          obj_category,
                          mode)

