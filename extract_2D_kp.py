import cv2


def extract_sift_kp_from_RGB(color_cropped, opt):
    gray = color_cropped[:, :, 0]
    sift = cv2.SIFT_create()
    kp_xys, des = sift.detectAndCompute(gray, None)
    color_sift = cv2.drawKeypoints(gray, kp_xys, color_cropped, (0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    return color_sift, kp_xys, des

