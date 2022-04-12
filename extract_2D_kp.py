import cv2


def extract_sift_kp_from_RGB(img_color_cropped, opt):
    img_gray = img_color_cropped[:, :, 0]
    sift = cv2.SIFT_create()
    kp_xys, des = sift.detectAndCompute(img_gray, None)
    color_sift = cv2.drawKeypoints(img_gray, kp_xys, img_color_cropped, (0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    return color_sift, kp_xys, des

