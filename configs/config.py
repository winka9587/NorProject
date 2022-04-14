import yaml
import torch
import os
from os.path import join as pjoin
sys.path.append('..')
from utils import ensure_dirs


def overwrite_config(cfg, key, key_path, value):
    cur_key = key_path[0]
    if len(key_path) == 1:
        old_value = None if cur_key not in cfg else cfg[cur_key]
        if old_value != value:
            print("{} (originally {}) overwritten by arg {}".format(key, old_value, value))
            cfg[cur_key] = value
    else:
        if not cur_key in cfg:
            cfg[cur_key] = {}
        overwrite_config(cfg[cur_key], key, key_path[1:], value)


def get_config(args, save=True):
    base_path = os.path.dirname(__file__)  # 当前config.py所在的目录(/CAPTRA/configs)为base_path
    f = open(pjoin(base_path, 'all_config', args.config), 'r')  # 去当前目录(/CAPTRA/configs)/all_config/--config路径下去找yml文件
    cfg = yaml.load(f, Loader=yaml.FullLoader)  # 读取all_config中跟踪的初始配置, 返回字典嵌套字典

    """ Update info from command line """
    # 使用启动时的命令行参数来替换初始配置中对应的配置
    args = vars(args)  # convert to a dictionary!
    args.pop('config')
    for key, item in args.items():
        if item is not None:
            key_path = key.split('/')
            overwrite_config(cfg, key, key_path, item)

    """ Load object config """
    f = open(pjoin(base_path, 'obj_config', cfg['obj_config']), 'r')
    obj_cfg = yaml.load(f, Loader=yaml.FullLoader)

    """ Load pointnet config """
    point_cfg_path = pjoin(base_path, 'pointnet_config')
    pointnet_cfgs = cfg['pointnet_cfg']
    cfg['pointnet'] = {}
    for key, value in pointnet_cfgs.items():
        f = open(pjoin(point_cfg_path, value), 'r')
        cfg['pointnet'][key] = yaml.load(f, Loader=yaml.FullLoader)

    """ Save config """
    root_dir = cfg['experiment_dir']
    ensure_dirs(root_dir)
    cfg['num_expr'] = root_dir.split('/')[-1]

    # 存储配置
    if save:
        yml_cfg = pjoin(root_dir, 'config.yml')
        yml_obj = pjoin(root_dir, cfg['obj_config'])
        print('Saving config and obj_config at {} and {}'.format(yml_cfg, yml_obj))
        # 这是在做什么？
        # zip将两个可迭代对象中的对应元素打包为元组，例如下面[(yml_cfg, cfg), (yml_obj, obj_cfg)]
        # 使用yaml.dump将cfg写入yml文件中
        for yml_file, content in zip([yml_cfg, yml_obj], [cfg, obj_cfg]):
            with open(yml_file, 'w') as f:
                yaml.dump(content, f, default_flow_style=False)

    # 将obj_cfg(obj_info_nocs.yml)中的一些配置添加到cfg中，用于一起返回
    """ Fill in additional info """
    obj_cat = cfg["obj_category"]
    cfg["num_parts"] = obj_cfg[obj_cat]["num_parts"]
    cfg["num_joints"] = obj_cfg[obj_cat]["num_joints"]
    cfg["obj_tree"] = obj_cfg[obj_cat]["tree"]
    cfg["obj_sym"] = obj_cfg[obj_cat]["sym"]  # sym是指什么？对称？
    cfg["obj"] = obj_cfg
    cfg["obj_info"] = obj_cfg[obj_cat]
    cfg["root_dset"] = obj_cfg['basepath']
    cfg["device"] = torch.device("cuda:%d" % cfg['cuda_id']) if torch.cuda.is_available() else "cpu"
    print("Running on ", cfg["device"])

    return cfg


# get base config like the path to load dataset, save result ...
def get_base_config():
    base_path = os.path.dirname(__file__)  # 当前config.py所在的目录(/CAPTRA/configs)为base_path
    # with open('base_config.yml', 'rb') as f:  # 去当前目录(/CAPTRA/configs)/all_config/--config路径下去找yml文件
    #     print(1)
    #     #base_cfg = yaml.load(f, Loader=yaml.FullLoader)
    base_cfg = {}
    return base_cfg
