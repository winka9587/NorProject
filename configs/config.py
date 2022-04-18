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
    base_path = os.path.dirname(__file__)  # ��ǰconfig.py���ڵ�Ŀ¼(/CAPTRA/configs)Ϊbase_path
    f = open(pjoin(base_path, 'all_config', args.config), 'r')  # ȥ��ǰĿ¼(/CAPTRA/configs)/all_config/--config·����ȥ��yml�ļ�
    cfg = yaml.load(f, Loader=yaml.FullLoader)  # ��ȡall_config�и��ٵĳ�ʼ����, �����ֵ�Ƕ���ֵ�

    """ Update info from command line """
    # ʹ������ʱ�������в������滻��ʼ�����ж�Ӧ������
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

    # �洢����
    if save:
        yml_cfg = pjoin(root_dir, 'config.yml')
        yml_obj = pjoin(root_dir, cfg['obj_config'])
        print('Saving config and obj_config at {} and {}'.format(yml_cfg, yml_obj))
        # ��������ʲô��
        # zip�������ɵ��������еĶ�ӦԪ�ش��ΪԪ�飬��������[(yml_cfg, cfg), (yml_obj, obj_cfg)]
        # ʹ��yaml.dump��cfgд��yml�ļ���
        for yml_file, content in zip([yml_cfg, yml_obj], [cfg, obj_cfg]):
            with open(yml_file, 'w') as f:
                yaml.dump(content, f, default_flow_style=False)

    # ��obj_cfg(obj_info_nocs.yml)�е�һЩ������ӵ�cfg�У�����һ�𷵻�
    """ Fill in additional info """
    obj_cat = cfg["obj_category"]
    cfg["num_parts"] = obj_cfg[obj_cat]["num_parts"]
    cfg["num_joints"] = obj_cfg[obj_cat]["num_joints"]
    cfg["obj_tree"] = obj_cfg[obj_cat]["tree"]
    cfg["obj_sym"] = obj_cfg[obj_cat]["sym"]  # sym��ָʲô���Գƣ�
    cfg["obj"] = obj_cfg
    cfg["obj_info"] = obj_cfg[obj_cat]
    cfg["root_dset"] = obj_cfg['basepath']
    cfg["device"] = torch.device("cuda:%d" % cfg['cuda_id']) if torch.cuda.is_available() else "cpu"
    print("Running on ", cfg["device"])

    return cfg


# get base config like the path to load dataset, save result ...
def get_base_config():
    base_path = os.path.dirname(__file__)  # ��ǰconfig.py���ڵ�Ŀ¼(/CAPTRA/configs)Ϊbase_path
    # with open('base_config.yml', 'rb') as f:  # ȥ��ǰĿ¼(/CAPTRA/configs)/all_config/--config·����ȥ��yml�ļ�
    #     print(1)
    #     #base_cfg = yaml.load(f, Loader=yaml.FullLoader)
    base_cfg = {}
    return base_cfg
