import cv2


def extract_sift_kp_from_RGB(img_color_cropped, opt=None):
    img_gray = cv2.cvtColor(img_color_cropped, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp_xys, des = sift.detectAndCompute(img_gray, None)
    color_sift = cv2.drawKeypoints(img_gray, kp_xys, img_color_cropped, (0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    return color_sift, kp_xys, des


def sift_match(des_1, des_2, matcher=None, k=2):
    if matcher:
        match_ = matcher
    else:
        match_ = cv2.BFMatcher(cv2.NORM_L2)
    matches = match_.knnMatch(des_1, des_2, k)
    return matches
