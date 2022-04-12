import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from utils import pjoin

def hist_test():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    x = [2500, 3000, 2700, 5600, 6700, 5400, 3100, 3500, 7600, 7800,
              8700, 9800, 10400]

    y = [1000, 1000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
    plt.hist(x)
    # plt.hist(x, y, histtype='bar', rwidth=0.8)
    plt.legend()
    plt.xlabel('salary-group')
    plt.ylabel('salary')
    plt.title(u'测试例子——直方图')

    plt.show()


# 如果要绘制曲率的直方图，应该给的x轴是曲率的大小区间，y轴是频数
def draw_histogram(array, arg_bin=10, name="hist", opt=None, clear_plt=True, ymax=7000):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

    plt.hist(array, bins=arg_bin, histtype='bar')
    plt.xlabel('curvature')
    plt.ylabel('num')
    plt.ylim(ymin=0, ymax=ymax)
    plt.title(u'curvature-histogram')
    if opt.vis_histogram:
        plt.show()
    if opt.save_root_path:
        plt_save_path = pjoin(opt.save_root_path, name+".png")
        plt.savefig(plt_save_path)
        print(f'plt img save to {plt_save_path}')
        if clear_plt:
            plt.clf()
    return plt

if __name__ == '__main__':
    hist_test()