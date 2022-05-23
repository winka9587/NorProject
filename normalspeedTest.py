# -*- coding: utf-8 -*-
import numpy as np
import time
import cv2
import os
import normalSpeed


def norm2bgr(norm):
    norm = ((norm + 1.0) * 127).astype("uint8")
    return norm


def depth2show(depth, norm_type='max'):
    show_depth = (depth / depth.max() * 256).astype("uint8")
    return show_depth


def generate_all(root_path, save_path):
    for file in os.listdir(root_path):
        if file[-10:] == '_depth.png':
            generate_single(root_path, save_path, file, viz=False)


def generate_single(root_path, save_root_path, depth_name, viz=True, save=True):
    # real
    fx = 591.0125
    fy = 590.16775
    # camera
    # fx = 577.5
    # fy = 577.5

    k_size = 1  # pixel
    distance_threshold = 50000  # mm

    point_into_surface = False
    scale = float(1.0)
    difference_threshold = 50 # mm

    # root_path = '/data1/cxx/Lab_work/dataset/nocs_full/real_train/scene_1/'
    # save_root_path = '/data1/cxx/normalSpeed-master/examplePicture/'
    # depth_name = '0001_depth.png'
    depth_path = os.path.join(root_path, depth_name)


    # color = np.load("examplePicture/0000_depth.npy")
    # depth = np.load("examplePicture/depth.npy")
    # depth = depth.astype(np.uint16)
    # depth = cv2.imread(depth_path, -1)[50:240, 150:380]
    depth = cv2.imread(depth_path, -1)
    for i in depth:
        for j in i:
            print(j, end=',')
        print('\n')
    print(f'depth:\n{depth}')
    # print(f'depth.dtype:\n{depth.dtype}')
    # depth = cv2.imread("examplePicture/0001_depth.png")[:,:,0]
    depth = depth * scale
    # print(f'depth: (float) {np.max(depth)},{np.min(depth)}')
    depth = depth.astype(np.uint16)
    # print(f'depth: (int)   {np.max(depth)},{np.min(depth)}')
    t1 = time.time()
    """
    The coordinate of depth and normal is in cv coordinate:
        - x is horizontal
        - y is down (to align to the actual pixel coordinates used in digital images)
        - right-handed: positive z look-at direction
    """
    normals_map_out = normalSpeed.depth_normal(depth, fx, fy, k_size, distance_threshold, difference_threshold,
                                               point_into_surface)

    # If the normal point out the surface, the z of normal should be negetive.
    print("normals_map_out z mean:", normals_map_out[:, :, 2].mean())
    # If the normal point into the surface, the z of normal should be positive, as z of depth.
    normals_map_in = normalSpeed.depth_normal(depth, fx, fy, k_size, distance_threshold, difference_threshold, True)
    print("normals_map_in z mean:", normals_map_in[:, :, 2].mean())

    info = f"_k={k_size}_dist={distance_threshold}_diff={difference_threshold}_scale_{scale}"

    if viz:
        cv2.imshow('norm_out', norm2bgr(normals_map_out))
        cv2.imshow('norm_in', norm2bgr(normals_map_in))
        cv2.imshow('depth', depth2show(depth))
        cv2.waitKey(10000)

    if save:
        cv2.imwrite(os.path.join(save_root_path, f"normal_out_{depth_name[-14:-10]}_{info}.png"), norm2bgr(normals_map_out))
        cv2.imwrite(os.path.join(save_root_path, f"normal_in_{depth_name[-14:-10]}_{info}.png"), norm2bgr(normals_map_in))
        cv2.imwrite(os.path.join(save_root_path, f"depth_view_{depth_name[-14:-10]}_{info}.png"), depth2show(depth))


if __name__ == "__main__":
    root_path = '/data1/cxx/Lab_work/dataset/nocs_full/real_train/scene_1/'
    save_root_path = '/data1/cxx/normalSpeed-master/examplePicture/'
    depth_name = '0425_depth.png'
    generate_single(root_path, save_root_path, depth_name, save=True)

    # root_path = '/data1/cxx/Lab_work/dataset/nocs_full/real_train/scene_1/'
    # folder_name = 'train_1'
    # save_root_path = os.path.join('/data1/cxx/normalmap/', folder_name)
    # generate_all(root_path, save_root_path)
