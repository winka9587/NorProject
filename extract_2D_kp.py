import cv2
import torch
import numpy as np

def extract_sift_kp_from_RGB(img_color_cropped, opt=None):
    if isinstance(img_color_cropped, torch.Tensor):
        img_color_cropped = img_color_cropped.clone().detach().cpu().numpy()
    img_gray = cv2.cvtColor(img_color_cropped, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp_xys, des = sift.detectAndCompute(img_gray, None)
    color_sift = cv2.drawKeypoints(img_gray, kp_xys, img_color_cropped, (0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    # color_sift = None
    return color_sift, kp_xys, des


def extract_sift_kp_from_gray(img_gray, opt=None):
    sift = cv2.SIFT_create()
    kp_xys, des = sift.detectAndCompute(img_gray, None)
    # color_sift = cv2.drawKeypoints(img_gray, kp_xys, img_color_cropped, (0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    color_sift = None
    return color_sift, kp_xys, des


def sift_match(des_1, des_2, matcher=None, k=2):
    if matcher:
        match_ = matcher
    else:
        match_ = cv2.BFMatcher(cv2.NORM_L2)
    matches = match_.knnMatch(des_1, des_2, k)
    return matches


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch, is_vis=True):
    points_1 = []
    points_2 = []
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
        points_1.append([x1, y1])
        points_2.append([x2, y2])
    # 可视化sift匹配结果
    if is_vis:
        cv2.namedWindow("match", cv2.WINDOW_NORMAL)
        cv2.imshow("match", vis)
        cv2.waitKey(0)

    match_points_1 = np.asarray(points_1)
    match_points_2 = np.asarray(points_2)
    return match_points_1, match_points_2-(w1, 0)