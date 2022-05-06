import open3d as o3d
import numpy as np
import cv2

def viz_multi_points_diff_color(name, pts_list, color_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=512, height=512, left=50, top=25)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    assert len(pts_list) == len(color_list)
    pcds = []
    for index in range(len(pts_list)):
        pcd = o3d.geometry.PointCloud()
        pts = pts_list[index]
        color = color_list[index]
        if np.any(color > 1.0):
            color = color.astype(float)/255.0
        colors = np.tile(color, (pts.shape[0], 1))  # 将color扩大为 (pts.shape[0], 1)
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcds.append(pcd)
        vis.add_geometry(pcd)
    ctr = vis.get_view_control()

    ctr.rotate(-300.0, 150.0)
    if name == 'camera':
        ctr.translate(20.0, -20.0)  # (horizontal right +, vertical down +)
    if name == 'laptop':
        ctr.translate(25.0, -60.0)
    vis.run()
    # if save_img and result_dir:
    #     vis.capture_screen_image(os.path.join(result_dir, name + '.png'), False)
    vis.destroy_window()


# 可视化mask
def viz_mask_bool(name, mask):
    mask_show = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    idx_ = np.where(mask)
    mask_show[idx_[0], idx_[1], :] = 255
    # cv2.imshow('rgb', raw_rgb)
    # cv2.waitKey(0)
    cv2.imshow(name, mask_show)
    cv2.waitKey(0)
