def test_coord_correspondence():
    # 测试coord图能否找到对应点
    # 在两幅coord图上连线对应点, xy1 tuple(x, y)
    def coord_line(img, img_p, xy1, xy2_s):
        # img_ = np.hstack((coord1, coord2))
        img_ = img.copy()
        img_p = img_p.copy()
        for xy2 in xy2_s:
            y2 = xy2[1]
            x2 = xy2[0]
            # x2 += img.shape[1]
            x2 += int(img.shape[1] / 2)
            cv2.line(img_, xy1, (x2, y2), (255, 0, 0), thickness=1)
            img_p[xy1[1], xy1[0], :] = 0
            img_p[xy1[1], xy1[0], 2] = 255
            img_p[y2, x2, :] = 0
            img_p[y2, x2, 2] = 255
        # cv2.imshow('coord respondence', img_)
        # cv2.waitKey(0)
        return img_, img_p

    prefix_1 = '0000'
    prefix_2 = '0040'
    target_instance_id_1 = 1  # 去找meta文件，想要哪个模型，就取其第一个值
    target_instance_id_2 = 5
    coord_1_path = f'/data1/cxx/Lab_work/dataset/nocs_full/real_train/scene_1/{prefix_1}_coord.png'
    coord_2_path = f'/data1/cxx/Lab_work/dataset/nocs_full/real_train/scene_1/{prefix_2}_coord.png'
    coord_1 = cv2.imread(coord_1_path)
    coord_2 = cv2.imread(coord_2_path)
    mask_1 = cv2.imread(coord_1_path.replace('coord', 'mask'))[:, :, 2]
    mask_2 = cv2.imread(coord_2_path.replace('coord', 'mask'))[:, :, 2]
    mask_1 = np.where(mask_1 == target_instance_id_1, True, False)
    mask_2 = np.where(mask_2 == target_instance_id_2, True, False)

    # coord_1 = data[0]['meta']['pre_fetched']['coord'].squeeze(0).numpy()
    # coord_2 = data[1]['meta']['pre_fetched']['coord'].squeeze(0).numpy()
    # mask_1 = data[0]['meta']['pre_fetched']['mask'].squeeze(0).numpy()
    # mask_2 = data[1]['meta']['pre_fetched']['mask'].squeeze(0).numpy()
    idx1 = np.where(mask_1 == False)
    idx2 = np.where(mask_2 == False)
    coord_1[idx1] = 0
    coord_2[idx2] = 0
    coord_c = np.hstack((coord_1, coord_2))
    coord_p = coord_c.copy()
    idx_non_zero_1 = coord_1.nonzero()
    color_0 = coord_2[:, :, 0].flatten()
    color_1 = coord_2[:, :, 1].flatten()
    color_2 = coord_2[:, :, 2].flatten()
    for idx_1 in range(len(idx_non_zero_1[0])):
        y1 = idx_non_zero_1[0][idx_1]
        x1 = idx_non_zero_1[1][idx_1]
        xy2_s = []
        # 遍历coord_1中的点,寻找早coord2中的对应点
        color = coord_1[y1, x1, :]
        print(f'searching {color}')
        _idx1 = np.where(color_0 == color[0])
        _idx2 = np.where(color_1 == color[1])
        _idx3 = np.where(color_2 == color[2])
        result_idx = reduce(np.intersect1d, [_idx1, _idx2, _idx3])
        if result_idx.shape[0] > 0:
            for e_idx in range(result_idx.shape[0]):
                idx_in_flatten = result_idx[e_idx]
                y = int(idx_in_flatten / 640)
                x = idx_in_flatten % 640
                print(f'idx:{e_idx}; pos:({y},{x}); value:{coord_2[y, x, :]} '
                      f'value2:{[color_0[idx_in_flatten], color_1[idx_in_flatten], color_2[idx_in_flatten]]}')
                xy2_s.append((x, y))
            coord_c = np.hstack((coord_1, coord_2))
            coord_c, coord_p = coord_line(coord_c, coord_p, (x1, y1), xy2_s)  # 可视化对应关系
    cv2.imshow('points', coord_p)
    cv2.waitKey(0)

# 测试新的求corr方法
def test_coord_correspondence2():
    # 测试coord图能否找到对应点
    # 在两幅coord图上连线对应点, xy1 tuple(x, y)
    def coord_line(img, img_p, xy1, xy2_s):
        # img_ = np.hstack((coord1, coord2))
        img_ = img.copy()
        img_p = img_p.copy()
        for xy2 in xy2_s:
            y2 = xy2[1]
            x2 = xy2[0]
            # x2 += img.shape[1]
            x2 += int(img.shape[1] / 2)
            cv2.line(img_, xy1, (x2, y2), (255, 0, 0), thickness=1)
            img_p[xy1[1], xy1[0], :] = 0
            img_p[xy1[1], xy1[0], 2] = 255
            img_p[y2, x2, :] = 0
            img_p[y2, x2, 2] = 255
        # cv2.imshow('coord respondence', img_)
        # cv2.waitKey(0)
        return img_, img_p

    # 在两张coord图上寻找颜色相同的点
    # coord已经用mask处理过
    def find_coord_correspondence(coord_1, coord_2):
        t = Timer(True)
        coord_1_flatten = coord_1.flatten().reshape(coord_1.shape[0] * coord_1.shape[1], -1)
        coord_2_flatten = coord_2.flatten().reshape(coord_2.shape[0] * coord_2.shape[1], -1)
        t.tick('flatten and reshape')
        # 能否在这里的时候将r和c一起append进去来节省时间？对应（1）的位置需要修改匹配
        coord_1_flatten_set = {(r, g, b) for [r, g, b] in coord_1_flatten if [r, g, b] != [0, 0, 0]}
        coord_2_flatten_set = {(r, g, b) for [r, g, b] in coord_2_flatten if [r, g, b] != [0, 0, 0]}
        t.tick('to set')
        coord_intersect_color = coord_1_flatten_set.intersection(coord_2_flatten_set)  # 取交集
        t.tick('intersection')
        corr_list = []
        for r, g, b in coord_intersect_color:
            # （1）这里也得修改
            r1, c1 = np.where((coord_1 == [r, g, b]).all(axis=-1))
            r2, c2 = np.where((coord_2 == [r, g, b]).all(axis=-1))
            assert len(r1) == 1 and len(c1) == 1 and len(r2) == 1 and len(c2) == 1
            corr_list.append((r1[0], c1[0], r2[0], c2[0]))
        t.tick('append to list')
        return corr_list

    # 返回的idx对应coord_1的坐标,可以直接通过coord_1[r,c]来进行读取
    def find_coord_correspondence2(coord_1, coord_2):
        print('==========new method==========')
        hash_1 = coord_1.copy()
        hash_2 = coord_2.copy()

        t = Timer(True)
        coord_1_flatten = coord_1.flatten().reshape(coord_1.shape[0] * coord_1.shape[1], -1)
        coord_2_flatten = coord_2.flatten().reshape(coord_2.shape[0] * coord_2.shape[1], -1)
        t.tick('flatten and reshape')
        # 能否在这里的时候将r和c一起append进去来节省时间？对应（1）的位置需要修改匹配
        # unique会丢失idx
        coord_1_flatten_set, idx_1 = np.unique(coord_1_flatten, return_index=True, axis=0)
        coord_2_flatten_set, idx_2 = np.unique(coord_2_flatten, return_index=True, axis=0)
        zero_idx1 = np.where((coord_1_flatten_set == [0, 0, 0]).all(axis=-1))[0][0]
        zero_idx2 = np.where((coord_2_flatten_set == [0, 0, 0]).all(axis=-1))[0][0]
        coord_1_flatten_set = np.delete(coord_1_flatten_set, [0, 0, 0], axis=0)
        coord_2_flatten_set = np.delete(coord_2_flatten_set, [0, 0, 0], axis=0)
        idx_1 = np.delete(idx_1, idx_1[zero_idx1], axis=0)
        idx_2 = np.delete(idx_2, idx_2[zero_idx2], axis=0)

        # coord_1_flatten_set = np.delete(, '[0, 0, 0]\n', axis=0)
        # coord_2_flatten_set = np.delete(np.unique(coord_2_flatten, axis=0), '[0, 0, 0]\n', axis=0)
        # 只能用set.intersection处理{tuple}
        # 不能用np.intersect1d来处理ndarray，即使ndarray中存的是tuple也不行，当ndarray超过一维，会被flatten
        # 一个方法，为每个color计算一个哈希值
        coord_1_flatten_set = [str(c) for c in coord_1_flatten_set]
        coord_2_flatten_set = [str(c) for c in coord_2_flatten_set]

        color_s, f_idx_1s, f_idx_2s = np.intersect1d(coord_1_flatten_set, coord_2_flatten_set, True, True)  # 取交集
        f_idx_1s = idx_1[f_idx_1s]
        f_idx_2s = idx_2[f_idx_2s]

        t.tick('intersection')
        corr_list = []
        w1 = coord_1.shape[1]
        w2 = coord_2.shape[1]
        for i in range(len(color_s)):
            r1 = int(f_idx_1s[i] / w1)
            c1 = f_idx_1s[i] % w1
            r2 = int(f_idx_2s[i] / w2)
            c2 = f_idx_2s[i] % w2
            corr_list.append((r1, c1, r2, c2))
        t.tick('append to list')
        return corr_list

    # 读取输入
    prefix_1 = '0000'
    prefix_2 = '0020'
    target_instance_id_1 = 1  # 去找meta文件，想要哪个模型，就取其第一个值
    target_instance_id_2 = 4
    coord_1_path = f'/data1/cxx/Lab_work/dataset/nocs_full/real_train/scene_1/{prefix_1}_coord.png'
    coord_2_path = f'/data1/cxx/Lab_work/dataset/nocs_full/real_train/scene_1/{prefix_2}_coord.png'
    coord_1 = cv2.imread(coord_1_path)
    coord_2 = cv2.imread(coord_2_path)
    mask_1 = cv2.imread(coord_1_path.replace('coord', 'mask'))[:, :, 2]
    mask_2 = cv2.imread(coord_2_path.replace('coord', 'mask'))[:, :, 2]
    mask_1 = np.where(mask_1 == target_instance_id_1, True, False)
    mask_2 = np.where(mask_2 == target_instance_id_2, True, False)

    # coord_1 = data[0]['meta']['pre_fetched']['coord'].squeeze(0).numpy()
    # coord_2 = data[1]['meta']['pre_fetched']['coord'].squeeze(0).numpy()
    # mask_1 = data[0]['meta']['pre_fetched']['mask'].squeeze(0).numpy()
    # mask_2 = data[1]['meta']['pre_fetched']['mask'].squeeze(0).numpy()
    idx1 = np.where(mask_1 == False)
    idx2 = np.where(mask_2 == False)
    coord_1[idx1] = 0
    coord_2[idx2] = 0

    # new
    timer = Timer(True)
    corr_list1 = find_coord_correspondence(coord_1, coord_2)
    print(f'len(corr_list): {len(corr_list1)}')
    timer.tick('find correspondence of two coord old method')
    corr_list2 = find_coord_correspondence2(coord_1, coord_2)
    print(f'len(corr_list): {len(corr_list2)}')
    timer.tick('find correspondence of two coord new method')
    coord_p0 = np.hstack((coord_1, coord_2))
    for r1, c1, r2, c2 in corr_list2:
        c2 += int(coord_1.shape[1])
        coord_p0[r1, c1, :] = 0
        coord_p0[r1, c1, 2] = 255
        coord_p0[r2, c2, :] = 0
        coord_p0[r2, c2, 2] = 255
    # cv2.imshow('new method', coord_p0)
    # cv2.waitKey(0)

    # old
    coord_c = np.hstack((coord_1, coord_2))
    coord_p = coord_c.copy()
    idx_non_zero_1 = coord_1.nonzero()
    color_0 = coord_2[:, :, 0].flatten()
    color_1 = coord_2[:, :, 1].flatten()
    color_2 = coord_2[:, :, 2].flatten()
    for idx_1 in range(len(idx_non_zero_1[0])):
        y1 = idx_non_zero_1[0][idx_1]
        x1 = idx_non_zero_1[1][idx_1]
        xy2_s = []
        # 遍历coord_1中的点,寻找早coord2中的对应点
        color = coord_1[y1, x1, :]
        # print(f'searching {color}')
        _idx1 = np.where(color_0 == color[0])
        _idx2 = np.where(color_1 == color[1])
        _idx3 = np.where(color_2 == color[2])
        result_idx = reduce(np.intersect1d, [_idx1, _idx2, _idx3])
        if result_idx.shape[0] > 0:
            for e_idx in range(result_idx.shape[0]):
                idx_in_flatten = result_idx[e_idx]
                y = int(idx_in_flatten / 640)
                x = idx_in_flatten % 640
                # print(f'idx:{e_idx}; pos:({y},{x}); value:{coord_2[y, x, :]} '
                #      f'value2:{[color_0[idx_in_flatten], color_1[idx_in_flatten], color_2[idx_in_flatten]]}')
                xy2_s.append((x, y))
            coord_c = np.hstack((coord_1, coord_2))
            # print('start draw')
            coord_c, coord_p = coord_line(coord_c, coord_p, (x1, y1), xy2_s)  # 可视化对应关系
    # cv2.imshow('points', coord_p)
    # cv2.waitKey(0)

    tmp = coord_p0 - coord_p
    print(f'if {len(np.where(tmp==0)[0])} == {coord_p.shape[0]*coord_p.shape[1]*coord_p.shape[2]} ? ')
