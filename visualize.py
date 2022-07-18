import open3d as o3d
import numpy as np
import cv2
import os

def RenderPcd(name, pts_list, color_list, show_coordinate=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=512, height=512, left=300, top=300)
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
    if show_coordinate:
        coordinateMesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        scale = 1.0
        coordinateMesh.scale(scale, center=(0, 0, 0))
        vis.add_geometry(coordinateMesh)

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

def RenderPcdNoHeader(name, pts_list, color_list, result_dir=None, show_coordinate=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=512, height=512, left=300, top=300, visible=False)
    # opt = vis.get_render_option()
    # opt.show_coordinate_frame = True
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

    if show_coordinate:
        # coordinateMesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # scale = 1.0
        # coordinateMesh.scale(scale, center=(0, 0, 0))
        # vis.add_geometry(coordinateMesh)

        # 手动绘制坐标系, 看看open3d的坐标系是否与实际一致
        coordLen = 1.25
        xyz = np.array([
            [0, 0, 0], [coordLen, 0, 0], [0, coordLen, 0], [0, 0, coordLen]
        ])
        lines = [
            [0, 1],
            [0, 2],
            [0, 3]
        ]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(xyz),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    ctr = vis.get_view_control()
    ctr.rotate(-300.0, 150.0)

    # vis.update_geometry() # 参数需要指定对应的geometry, 但是没有修改任何geometry，跟不需要update
    # vis.poll_events()
    vis.update_renderer()

    # vis.run()
    if result_dir:
        savePath = os.path.join(result_dir, name + '.png')
        vis.capture_screen_image(savePath, False)
        print(f'[RenderPcdNoHeader][OK] Save Img to {savePath}')
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
