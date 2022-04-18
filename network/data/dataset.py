import math
import numpy as np
import torch.utils.data as data
from os.path import join as pjoin
import cv2
from utils import ensure_dir
from captra_utils.utils_from_captra import load_depth
import _pickle as cPickle

# result_path:
# result_path:
#       --img_list
# mode: [train, test]
# source: [Real, CAMERA, Real_CAMERA]
class NOCSDataset(data.Dataset):
    def __init__(self, dataset_path, result_path, obj_category, mode, source, n_points=4096, augment=False):
        assert (mode == 'train' or mode == 'val'), 'Mode must be "train" or "val".'
        self.dataset_path = dataset_path
        self.result_path = result_path
        self.obj_category = obj_category
        self.mode = mode
        self.n_points = n_points
        self.augment = augment

        # augmentation parameters
        self.sigma = 0.01
        self.clip = 0.02
        self.shift_range = 0.02

        # img_list
        img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
                           'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']
        # ����ʹ�õ����ݼ���ѡ���_path��ɾȥ��Щtxt��pkl�ļ�
        if mode == 'train':
            del img_list_path[2:]
            del model_file_path[2:]
        else:
            del img_list_path[:2]
            del model_file_path[:2]

        if source == 'CAMERA':
            del img_list_path[-1]
            del model_file_path[-1]
        elif source == 'Real':
            del img_list_path[0]
            del model_file_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del img_list_path[0]
                del model_file_path[0]
        img_list = []  # �洢��img_list�µ�txt�ж�ȡ��ǰ׺·��
        subset_len = []  # �洢txt�е��ļ������������з�img_list����
        for path in img_list_path:
            # ��ȡ_list.txt�ļ�,���е�ÿһ�ж���train/0000/0000�ĸ�ʽ
            # ȥ�����з�,path.splitȡCAMERA
            # ���õ�CAMERA/train/0000/0000ģ���·������img_list��
            img_list += [pjoin(path.split('/')[0], line.rstrip('\n'))
                         for line in open(pjoin(self.result_path, 'img_list', path))]
            subset_len.append(len(img_list))
        # ��Ϊsource��modeɸѡ��ԭ��,����img_list_path��ʣ�µ�ֻ�п�����1����2��
        # ���ֻ��1��,��ôappend��ӵľ���length
        # ���������,��Ϊimg_list�ǲ���յ�,������Ҫ��2��-��1��
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1] - subset_len[0]]
        self.img_list = img_list
        self.length = len(self.img_list)

        # ��ȡ����
        models = {}
        for path in model_file_path:
            with open(pjoin(self.dataset_path, path), 'rb') as f:
                models.update(cPickle.load(f))  # update��һ�����ϲ�����һ������
        self.models = models

        # meta info for re-label mug category
        with open(pjoin(self.dataset_path, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        # ��ȡ��ȡembedding
        # self.mean_shapes = np.load('assets/mean_points_emb.npy')
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.camera_intrinsics = [577.5, 577.5, 319.5, 239.5]  # [fx, fy, cx, cy]
        self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        self.sym_ids = [0, 1, 3]  # 0-indexed
        self.norm_scale = 1000.0  # normalization scale
        # ���ɵ�xmap
        # [
        # [ 0,1,2,...],
        # [ 0,1,2,...],
        # ...]
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.shift_range = 0.01
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        print('{} images found.'.format(self.length))
        print('{} models loaded.'.format(len(self.models)))

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
