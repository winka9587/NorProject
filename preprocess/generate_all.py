# 运行generate_all.py,会根据命令行参数(数据集路径)生成一个nocs_preproc_all.sh来保存命令
# 之后会运行bash nocs_preproc_all.sh来执行这些命令
# 同目录下的script.sh是个示例

import os
import sys
import argparse
from os.path import join as pjoin

# 数据集解压完成后运行当前文件来进行数据集的预处理
#   cd CAPTRA/datasets/nocs_data/preproc_nocs
#   python generate_all.py
#           --data_path ../../../../data/nocs_data      数据集所在路径
#           --data_type=all                             处理哪些数据:all处理所有;test_only仅处理real_test
#           --parallel
#           --num_proc=10
#           #(这个才是关键,将print的结果都保存到.sh文件中)
#           > nocs_preproc_all.sh # generate the script for data preprocessing
#   # parallel & num_proc specifies the number of parallel processes in the following procedure
#   bash nocs_preproc_all.sh # the actual data preprocessing
def main(root_dset='../../../../data/nocs_data', data_type='all', parallel=True, num_proc=10):

    def execute(s):
        print(s)
        # os.system(s)

    # 默认的all_data是['train', 'val', 'real_train', 'real_test'] 4个文件夹都有的
    # --data_path='data/nocs_data/nocs_full'
    syn_data = ['train', 'val']  # syn指合成数据
    all_data = ['train', 'val', 'real_train', 'real_test']
    if data_type == 'test_only':
        syn_data = []
        all_data = ['real_test']

    categories = list(range(1, 7))  # 1,2,3,4,5,6共6个类别的数据

    root_dset = os.path.abspath(root_dset)  # 数据集的绝对路径(至/data/nocs_data)

    ori_path = pjoin(root_dset, 'nocs_full')  # nocs数据集路径 /data/nocs_data/nocs_full
    ikea_path = pjoin(root_dset, 'ikea_data')
    list_path = pjoin(root_dset, 'instance_list')
    model_path = pjoin(root_dset, 'model_corners')  # 这个路径做什么用的？
    output_path = pjoin(root_dset, 'render')

    parallel_suffix = '' if not parallel else f' --parallel --num_proc={num_proc}'

    """
    for data_type in syn_data:
        cmd = 'python match_table.py' + \
              f' --data_path={pjoin(ori_path, data_type)} --bg_path={ikea_path}' + \
              parallel_suffix
        execute(cmd)
    """
    # windows下需要对命令中的路径加引号
    windows = False
    # 获取每个实例的gt位姿
    for data_type in all_data:
        if windows:
            cmd = 'python get_gt_poses.py' + \
                  f' --data_path=\'{ori_path}\' --data_type=\'{data_type}\'' + \
                  parallel_suffix
        else:
            cmd = 'python get_gt_poses.py' + \
              f' --data_path={ori_path} --data_type={data_type}' + \
              parallel_suffix
        execute(cmd)

    # 获取实例列表
    for data_type in all_data:
        if windows:
            cmd = 'python get_instance_list.py' + \
                  f' --data_path=\'{ori_path}\' --data_type=\'{data_type}\' --list_path=\'{list_path}\'' + \
                  parallel_suffix
        else:
            cmd = 'python get_instance_list.py' + \
                  f' --data_path={ori_path} --data_type={data_type} --list_path={list_path}' + \
                  parallel_suffix
        execute(cmd)

    # 收集实例数据
    for data_type in all_data:
        for category in categories:
            if windows:
                cmd = 'python gather_instance_data.py' + \
                      f' --data_path=\'{ori_path}\' --data_type=\'{data_type}\' --list_path=\'{list_path}\'' + \
                      f' --model_path=\'{model_path}\' --output_path=\'{output_path}\' --category=\'{category}\'' + \
                      parallel_suffix
            else:
                cmd = 'python gather_instance_data.py' + \
                    f' --data_path={ori_path} --data_type={data_type} --list_path={list_path}' + \
                    f' --model_path={model_path} --output_path={output_path} --category={category}' + \
                    parallel_suffix
            execute(cmd)

    # 构建链接
    if 'val' in all_data:
        if windows:
            execute(f'ln -s \'{pjoin(output_path, "val")}\' \'{pjoin(output_path, "test")}\'')
        else:
            execute(f'ln -s {pjoin(output_path, "val")} {pjoin(output_path, "test")}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../../../nocs_data')
    parser.add_argument('--data_type', type=str, default='all')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_proc', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.data_type in ['all', 'test_only']
    # parallel和num_proc指定了接下来过程中的并行进程数量
    # python generate_all.py
    #                       --data_path ../../../../data/nocs_data
    #                       --data_type=test_only
    # 			            --parallel  # store_true 只要带着这个参数，其值就为True
    # 			            --num_proc=10
    # 			            > nocs_preproc.sh  # 结果输出到.sh文件中
    main(args.data_path, args.data_type, args.parallel, args.num_proc)
