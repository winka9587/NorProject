#coding=gbk
import os
import sys
import glob
import cv2
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
from align import align_nocs_to_depth
from preprocess_utils import load_depth
sys.path.append('../configs/')
from configs.base_config import get_base_config
from os.path import join as pjoin
from utils import ensure_dir


def create_img_list(data_dir, save_dir):
    save_dir = pjoin(save_dir, 'img_list')
    """ Create train/val/test data list for CAMERA and Real. """
    # CAMERA dataset
    # ��data/CAMERAĿ¼�£�����train_list_all.txt��val_list_all.txt����¼���г�����·��
    for subset in ['train', 'val']:
        img_list = []
        img_dir = os.path.join(data_dir, 'CAMERA', subset)
        # folder_list��¼train��valĿ¼�������ļ�����
        folder_list = [name for name in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, name))]
        # һ���ļ�������10�����������ݣ�һ��������4���ļ���_color(��ɫͼ),_mask(����),_coord(NOCSͼ),_meta.txt(δ֪)
        # ������һ��img_list������train��val�µ�����ͼƬ·�� img_path ����¼����
        # Ϊ����֮�󷽱���img_pathȥ��ȡ�Ƕ�Ӧ��4���ļ�
        for i in range(10*len(folder_list)):
            folder_id = int(i) // 10
            img_id = int(i) % 10
            # ����:train/27498/0000
            img_path = os.path.join(subset, '{:05d}'.format(folder_id), '{:04d}'.format(img_id))
            img_list.append(img_path)
        ensure_dir(pjoin(save_dir, 'CAMERA'), True)
        with open(os.path.join(save_dir, 'CAMERA', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
        print('create_img_list {}'.format(os.path.join(save_dir, 'CAMERA', subset+'_list_all.txt')))
    # Real dataset
    # Real���ݼ���CAMERA���ݼ��ṹ��ͬ
    for subset in ['train', 'test']:
        img_list = []
        img_dir = os.path.join(data_dir, 'Real', subset)
        # folder_list=[scene_1,scene_2,...]
        folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
        for folder in folder_list:
            img_paths = glob.glob(os.path.join(img_dir, folder, '*_color.png'))
            img_paths = sorted(img_paths)
            for img_full_path in img_paths:
                # ����img_full_path=/xxx/0000_color.png
                img_name = os.path.basename(img_full_path)
                img_ind = img_name.split('_')[0]
                img_path = os.path.join(subset, folder, img_ind)
                # img_path=/xxx/0000
                img_list.append(img_path)
        ensure_dir(pjoin(save_dir, 'Real'), True)
        with open(os.path.join(save_dir, 'Real', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
        print('create_img_list {}'.format(os.path.join(save_dir, 'Real', subset + '_list_all.txt')))
    print('Write all data paths to file done!')


def process_data(img_path, depth):
    """ Load instance masks for the objects in the image. """
    # ��ȡmaskͼ
    mask_path = img_path + '_mask.png'
    mask = cv2.imread(mask_path)[:, :, 2]  # maskֻ��Ҫһ��ͨ��
    mask = np.array(mask, dtype=np.int32)
    # unique�����������ǽ�mask�е�����������¼����(�ظ���ɾ��)
    # ����mask��ֻ��ֵΪ0,20,255������,���շ���(0,20,255)
    all_inst_ids = sorted(list(np.unique(mask)))
    assert all_inst_ids[-1] == 255
    del all_inst_ids[-1]    # remove background
    # ������ɾ����
    num_all_inst = len(all_inst_ids)
    h, w = mask.shape

    # ��ȡnocsͼ
    coord_path = img_path + '_coord.png'
    coord_map = cv2.imread(coord_path)[:, :, :3]
    # RGB is actually BGR in opencv, ����Ҫ����ά��
    coord_map = coord_map[:, :, (2, 1, 0)]
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    # num_all_inst��mask��ֵ������
    class_ids = []
    instance_ids = []
    model_list = []
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)

    meta_path = img_path + '_meta.txt'
    # ��ÿһ�ж�Ӧ��ʵ�����д���
    with open(meta_path, 'r') as f:
        i = 0
        for line in f:
            # line_info=[inst_id, cls_id, mug2_scene3_norm]
            line_info = line.strip().split(' ')
            inst_id = int(line_info[0])  # ʵ��id
            cls_id = int(line_info[1])  # ��id
            # background objects and non-existing objects
            # ���ǩcls_idΪ0�ǲ����ڵ�����
            if cls_id == 0 or (inst_id not in all_inst_ids):
                continue
            if len(line_info) == 3:
                model_id = line_info[2]    # Real scanned objs
            else:
                model_id = line_info[3]    # CAMERA objs
            # remove one mug instance in CAMERA train due to improper model
            if model_id == 'b9be7cfe653740eb7633a2dd89cec754':
                continue
            # process foreground objects
            # ǰ������(ʵ��inst_mask)��Ӧ������ΪTrue,������ΪFalse
            inst_mask = np.equal(mask, inst_id)
            # bounding box
            # np.any ��ĳһ����л�����,�������True�򷵻�True
            # Ȼ��ʹ��np.where,��¼һά����������ΪTrue�ط���index
            # �������ֱܷ�õ�ˮƽ��ֱ������ֵ��������
            horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
            vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
            assert horizontal_indicies.shape[0], print(img_path)
            # �������˵�����,�����򶨱߽�
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            # ��
            x2 += 1
            y2 += 1
            # object occupies full image, rendering error, happens in CAMERA dataset
            if np.any(np.logical_or((x2-x1) > 600, (y2-y1) > 440)):
                return None, None, None, None, None, None
            # not enough valid depth observation
            final_mask = np.logical_and(inst_mask, depth > 0)
            if np.sum(final_mask) < 64:
                continue
            class_ids.append(cls_id)
            instance_ids.append(inst_id)
            model_list.append(model_id)
            masks[:, :, i] = inst_mask  # True
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            bboxes[i] = np.array([y1, x1, y2, x2])
            i += 1
    # no valid foreground objects
    if i == 0:
        return None, None, None, None, None, None
    # masks,coords,bboxes������ά����num_all_inst,ÿһ���Ӧ_meta.txt�е�һ��ʵ��
    masks = masks[:, :, :i]
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]

    return masks, coords, class_ids, instance_ids, model_list, bboxes


# ��ȡimg_list_dir(img_list�ļ���)�µ�train_list_all.txt
# �������ܹ���Ч����gt��ǰ׺����ͬĿ¼�µ�train_list.txt
# �������gt(.pkl�ļ�)������img_listͬĿ¼��gt_label��
def annotate_camera_train(dataset_dir, save_path):
    img_list_dir = pjoin(save_path, 'img_list')
    gt_label_dir = pjoin(save_path, 'gt_label')
    """ Generate gt labels for CAMERA train data. """
    camera_train = open(os.path.join(img_list_dir, 'CAMERA', 'train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    # meta info for re-label mug category
    with open(os.path.join(dataset_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    valid_img_list = []
    for img_path in tqdm(camera_train):
        img_full_path = os.path.join(dataset_dir, 'CAMERA', img_path)
        gt_label_path = pjoin(gt_label_dir, 'CAMERA', img_path)
        ensure_dir(pjoin(gt_label_dir, 'CAMERA'), True)
        # ����CAMERA���ݼ��µ�·���Ƿ����ȫ��5���ļ�
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        # ��ȡ���������ͼ
        depth = load_depth(img_full_path)
        # ���ͼ�������е�mask,nocsͼ,��id,ʵ��id,model_list(mug2_scene3_norm,mug2_scene3_norm...),2D��Χ��
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        # Umeyama alignment of GT NOCS map with depth image
        # ���NOCSͼ�����ͼ,Umeyama�㷨,�õ�λ��sRT
        scales, rotations, translations, error_messages, _ = \
            align_nocs_to_depth(masks, coords, depth, intrinsics, instance_ids, img_path)
        if error_messages:
            continue
        # re-label for mug category
        for i in range(len(class_ids)):
            if class_ids[i] == 6:
                T0 = mug_meta[model_list[i]][0]
                s0 = mug_meta[model_list[i]][1]
                T = translations[i] - scales[i] * rotations[i] @ T0
                s = scales[i] / s0
                scales[i] = s
                translations[i] = T
        # write results
        gts = {}
        gts['class_ids'] = class_ids    # int list, 1 to 6
        gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)    # np.array, R
        gts['translations'] = translations.astype(np.float32)  # np.array, T
        gts['instance_ids'] = instance_ids  # int list, start from 1
        gts['model_list'] = model_list  # str list, model id/name
        ensure_dir(gt_label_path[:(max(gt_label_path.rfind('/'), gt_label_path.rfind('\\')))], True)
        with open(gt_label_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
    # write valid img list to file
    with open(os.path.join(img_list_dir, 'CAMERA/train_list.txt'), 'w') as f:
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)


def annotate_real_train(dataset_dir, save_path):
    img_list_dir = pjoin(save_path, 'img_list')
    gt_label_dir = pjoin(save_path, 'gt_label')
    """ Generate gt labels for Real train data through PnP. """
    real_train = open(os.path.join(img_list_dir, 'Real/train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # scale factors for all instances
    scale_factors = {}
    path_to_size = glob.glob(os.path.join(dataset_dir, 'obj_models/real_train', '*_norm.txt'))
    for inst_path in sorted(path_to_size):
        instance = os.path.basename(inst_path).split('.')[0]  # instance model name (like: mug_white_green_norm)
        bbox_dims = np.loadtxt(inst_path)  # read bbox txt
        scale_factors[instance] = np.linalg.norm(bbox_dims)  # Ȼ��ֱ�Ӽ���ģ,��Ϊscale
    # meta info for re-label mug category
    with open(os.path.join(dataset_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    valid_img_list = []
    for img_path in tqdm(real_train):
        img_full_path = os.path.join(dataset_dir, 'Real', img_path)
        gt_label_path = pjoin(gt_label_dir, 'Real', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        depth = load_depth(img_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        # compute pose
        num_insts = len(class_ids)
        scales = np.zeros(num_insts)
        rotations = np.zeros((num_insts, 3, 3))
        translations = np.zeros((num_insts, 3))
        for i in range(num_insts):
            s = scale_factors[model_list[i]]
            mask = masks[:, :, i]
            idxs = np.where(mask)
            coord = coords[:, :, i, :]
            coord_pts = s * (coord[idxs[0], idxs[1], :] - 0.5)
            coord_pts = coord_pts[:, :, None]
            img_pts = np.array([idxs[1], idxs[0]]).transpose()
            img_pts = img_pts[:, :, None].astype(float)
            distCoeffs = np.zeros((4, 1))    # no distoration
            retval, rvec, tvec = cv2.solvePnP(coord_pts, img_pts, intrinsics, distCoeffs)
            assert retval
            R, _ = cv2.Rodrigues(rvec)
            T = np.squeeze(tvec)
            # re-label for mug category
            if class_ids[i] == 6:
                T0 = mug_meta[model_list[i]][0]
                s0 = mug_meta[model_list[i]][1]
                T = T - s * R @ T0
                s = s / s0
            scales[i] = s
            rotations[i] = R
            translations[i] = T
        # write results
        gts = {}
        gts['class_ids'] = class_ids    # int list, 1 to 6
        gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)    # np.array, R
        gts['translations'] = translations.astype(np.float32)  # np.array, T
        gts['instance_ids'] = instance_ids  # int list, start from 1
        gts['model_list'] = model_list  # str list, model id/name
        ensure_dir(gt_label_path[:(max(gt_label_path.rfind('/'), gt_label_path.rfind('\\')))], True)
        with open(gt_label_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
    # write valid img list to file
    with open(os.path.join(img_list_dir, 'Real/train_list.txt'), 'w') as f:
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)


def annotate_test_data(dataset_dir, save_path):
    img_list_dir = pjoin(save_path, 'img_list')
    gt_label_dir = pjoin(save_path, 'gt_label')
    """ Generate gt labels for test data.
        Properly copy handle_visibility provided by NOCS gts.
    """
    # Statistics:
    # test_set    missing file     bad rendering    no (occluded) fg    occlusion (< 64 pts)
    #   val        3792 imgs        132 imgs         1856 (23) imgs      50 insts
    #   test       0 img            0 img            0 img               2 insts

    camera_val = open(os.path.join(img_list_dir, 'CAMERA', 'val_list_all.txt')).read().splitlines()
    real_test = open(os.path.join(img_list_dir, 'Real', 'test_list_all.txt')).read().splitlines()
    camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # compute model size
    model_file_path = ['obj_models/camera_val.pkl', 'obj_models/real_test.pkl']
    models = {}
    for path in model_file_path:
        with open(os.path.join(dataset_dir, path), 'rb') as f:
            models.update(cPickle.load(f))
    model_sizes = {}
    for key in models.keys():
        model_sizes[key] = 2 * np.amax(np.abs(models[key]), axis=0)
    # meta info for re-label mug category
    with open(os.path.join(dataset_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    subset_meta = [('CAMERA', camera_val, camera_intrinsics, 'val'), ('Real', real_test, real_intrinsics, 'test')]
    for source, img_list, intrinsics, subset in subset_meta:
        valid_img_list = []
        for img_path in tqdm(img_list):
            img_full_path = os.path.join(dataset_dir, source, img_path)
            gt_label_path = pjoin(gt_label_dir, source, img_path)
            all_exist = os.path.exists(img_full_path + '_color.png') and \
                        os.path.exists(img_full_path + '_coord.png') and \
                        os.path.exists(img_full_path + '_depth.png') and \
                        os.path.exists(img_full_path + '_mask.png') and \
                        os.path.exists(img_full_path + '_meta.txt')
            if not all_exist:
                continue
            depth = load_depth(img_full_path)
            masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
            if instance_ids is None:
                continue
            num_insts = len(instance_ids)
            # match each instance with NOCS ground truth to properly assign gt_handle_visibility
            nocs_dir = os.path.join(dataset_dir, 'gts')
            if source == 'CAMERA':
                nocs_path = os.path.join(nocs_dir, 'val', 'results_val_{}_{}.pkl'.format(
                    img_path.split('/')[-2], img_path.split('/')[-1]))
            else:
                nocs_path = os.path.join(nocs_dir, 'real_test', 'results_test_{}_{}.pkl'.format(
                    img_path.split('/')[-2], img_path.split('/')[-1]))
            with open(nocs_path, 'rb') as f:
                nocs = cPickle.load(f)
            gt_class_ids = nocs['gt_class_ids']
            gt_bboxes = nocs['gt_bboxes']
            gt_sRT = nocs['gt_RTs']
            gt_handle_visibility = nocs['gt_handle_visibility']
            map_to_nocs = []
            for i in range(num_insts):
                gt_match = -1
                for j in range(len(gt_class_ids)):
                    if gt_class_ids[j] != class_ids[i]:
                        continue
                    if np.sum(np.abs(bboxes[i] - gt_bboxes[j])) > 5:
                        continue
                    # match found
                    gt_match = j
                    break
                # check match validity
                assert gt_match > -1, print(img_path, instance_ids[i], 'no match for instance')
                assert gt_match not in map_to_nocs, print(img_path, instance_ids[i], 'duplicate match')
                map_to_nocs.append(gt_match)
            # copy from ground truth, re-label for mug category
            handle_visibility = gt_handle_visibility[map_to_nocs]
            sizes = np.zeros((num_insts, 3))
            poses = np.zeros((num_insts, 4, 4))
            scales = np.zeros(num_insts)
            rotations = np.zeros((num_insts, 3, 3))
            translations = np.zeros((num_insts, 3))
            for i in range(num_insts):
                gt_idx = map_to_nocs[i]
                sizes[i] = model_sizes[model_list[i]]
                sRT = gt_sRT[gt_idx]
                s = np.cbrt(np.linalg.det(sRT[:3, :3]))
                R = sRT[:3, :3] / s
                T = sRT[:3, 3]
                # re-label mug category
                if class_ids[i] == 6:
                    T0 = mug_meta[model_list[i]][0]
                    s0 = mug_meta[model_list[i]][1]
                    T = T - s * R @ T0
                    s = s / s0
                # used for test during training
                scales[i] = s
                rotations[i] = R
                translations[i] = T
                # used for evaluation
                sRT = np.identity(4, dtype=np.float32)
                sRT[:3, :3] = s * R
                sRT[:3, 3] = T
                poses[i] = sRT
            # write results
            gts = {}
            gts['class_ids'] = np.array(class_ids)    # int list, 1 to 6
            gts['bboxes'] = bboxes    # np.array, [[y1, x1, y2, x2], ...]
            gts['instance_ids'] = instance_ids    # int list, start from 1
            gts['model_list'] = model_list    # str list, model id/name
            gts['size'] = sizes   # 3D size of NOCS model
            gts['scales'] = scales.astype(np.float32)    # np.array, scale factor from NOCS model to depth observation
            gts['rotations'] = rotations.astype(np.float32)    # np.array, R
            gts['translations'] = translations.astype(np.float32)    # np.array, T
            gts['poses'] = poses.astype(np.float32)    # np.array
            gts['handle_visibility'] = handle_visibility    # handle visibility of mug
            with open(img_full_path + '_label.pkl', 'wb') as f:
                cPickle.dump(gts, f)
            valid_img_list.append(img_path)
        # write valid img list to file
        with open(os.path.join(data_dir, source, subset+'_list.txt'), 'w') as f:
            for img_path in valid_img_list:
                f.write("%s\n" % img_path)


if __name__ == '__main__':
    base_cfg = get_base_config()
    print(base_cfg['dataset_path'])
    print(base_cfg['result_path'])
    dataset_path = base_cfg['dataset_path']
    save_path = base_cfg['result_path']
    # create list for all data
    create_img_list(dataset_path, save_path)
    # annotate dataset and re-write valid data to list
    # annotate_camera_train(dataset_path, save_path)
    annotate_real_train(dataset_path, save_path)
    # annotate_test_data(data_dir)
