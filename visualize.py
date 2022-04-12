import open3d as o3d
import numpy as np


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
        colors = np.tile(color, (pts.shape[0], 1))  # ½«colorÀ©´óÎª (pts.shape[0], 1)
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
