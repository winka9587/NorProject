# 前背景分割网络

# 输入:
# pts1: masked_points_1 (1024 x 3)
# pts2: (4096 x 3)

# 1.提取pts2几何特征 (4096 x 64)
# 2.提取pts2全局特征 1x1024
# 2.提取pts1的全局特征1x1024, concat到pts2的点特征后面, (4096 x 64) + 4096x(1x1024)x2 -> (4096 x 64+1024+1024)
# 分割

