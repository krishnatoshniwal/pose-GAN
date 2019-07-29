from easydict import EasyDict as edict
import numpy as np
import proj_conf
from resources.utils import special_string, ensure_dir_exists, save_obj
import datetime
import os.path as osp
import os
import distutils.dir_util as dir_util

cfg = edict()
cfg.flags = edict()
cfg.dirs = edict()
cfg.out_files = edict()

cfg.model_param = edict()
cfg.train_param = edict()
cfg.sample_param = edict()
cfg.optimizer_param = edict()
cfg.log_param = edict()
cfg.params = edict()

""" Config parameters """
# Output files
cfg.dirs.run_name = 'default_name'
# cfg.flags.SAVE_RUN = True

# Optimizer Params
cfg.optimizer_param.name = 'Adam'
cfg.optimizer_param.g_lr = 0.0002
cfg.optimizer_param.d_lr = 0.0005
cfg.train_param.critic_train_no = 1

cfg.optimizer_param.lr_decay_rate = 0.9
cfg.optimizer_param.beta1 = 0.5
cfg.optimizer_param.beta2 = 0.999
# cfg.optimizer_param.weight_decay = 0.0005

# Training Params
cfg.train_param.num_workers = 4
cfg.train_param.batch_size = 20

cfg.train_param.global_iter = 0
cfg.train_param.max_iters = 200000

cfg.train_param.model_save_step = 15000
cfg.train_param.sample_step = 500
cfg.train_param.log_step = 100
cfg.train_param.lr_update_step = 50000
cfg.train_param.poses = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

# cfg.train_param.INCLUDE_INTERP_LOSS = True
cfg.train_param.LOAD_NUMBER_POSE = False
cfg.train_param.INTERPOLATION_MODE = False
cfg.train_param.scale_fact_pose = 100.0

# Model Parameters
cfg.model_param.gen = "gena"
cfg.model_param.dis = "sn"
cfg.model_param.crop_size = 480
cfg.model_param.img_size = 128
cfg.model_param.d_conv_dim = 32

# cfg.model_param.g_repeat_num = 6

cfg.model_param.lambda_gp = 10
cfg.model_param.lambda_cls = 1
cfg.model_param.lambda_tvr = 1e-6
cfg.model_param.lambda_l1 = 5
cfg.model_param.lambda_cyc = 1
cfg.model_param.lambda_att = 1
cfg.model_param.lambda_interpol = 0.5

cfg.sample_param.num_samples = 10
cfg.sample_param.interp_poses = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
cfg.log_param.LOG_IMAGES = False

# Flags, seed and other parameters
cfg.flags.LOAD_FROM_CONFIG = False
cfg.flags.IS_CUDA = True
cfg.flags.BUILD_IMDB = False
cfg.flags.LOADED_CONFIG = False
cfg.flags.TEST_BUILD = False
cfg.flags.APPEND_DATETIME = False
cfg.flags.USE_TENSORBOARD = True
cfg.flags.DATA_PARALLEL = 0

cfg.params.cuda_device_id = 'cuda:0'
cfg.params.rand_seed = None
cfg.params.parallel_device_id = '1,2,3,4'
cfg.params.run_info = "..."  # info about the current run

# Directories/Paths
cfg.dirs.project_root = proj_conf.project_root
cfg.dirs.imdb_loc = proj_conf.imdb_loc
cfg.dirs.dataset_dir = proj_conf.dataset_dir
cfg.dirs.pose_img_dir = proj_conf.pose_imgs_dir
cfg.dirs.saved_data_loc = proj_conf.save_data_loc
cfg.dirs.landmarks_file = proj_conf.landmarks_file
cfg.dirs.save_dir = 0

# cfg.dirs.lcnn_path = proj_conf.lCNN_path

cfg.dirs.log_dir = 'logs'
cfg.dirs.model_save_dir = 'models'
cfg.dirs.result_dir = 'results'
cfg.dirs.sample_dir = 'samples'
cfg.dirs.interpolate_dir = 'interpolated'
cfg.dirs.export_code = 'code'

cfg.out_files.config_file_loc = 'config.pkl'
cfg.out_files.config_text_loc = 'config.txt'


# parser.add_argument('--model_restore_dir', type=str, default='hq/models')

def export_files(conf):
    dir_util.copy_tree(os.getcwd(), conf.dirs.export_code)
    dir_util.remove_tree(osp.join(conf.dirs.export_code, '.git'))


def arg2conf_alias():
    """
    The default alias set for the most used variables
    """
    arg2conf_dict = {
        'use_tensorboard': ['flags', 'USE_TENSORBOARD'],
        'load_dir': ['dirs', 'save_dir'],
        'cuda': ['params', 'cuda_device_id'],
        'n_critic': ['train_param', 'critic_train_no'],
        'bs': ['train_param', 'batch_size'],
        'test_build': ['flags', 'TEST_BUILD'],
        'info': ['params', 'run_info'],
        'interp': ['train_param', 'INTERPOLATION_MODE'],
        'numerical': ['train_param', 'LOAD_NUMBER_POSE']
    }
    return arg2conf_dict


def initialise_config(conf):
    """
    Sets up conf dictionary
    """

    if conf.params.rand_seed is None:
        conf.params.rand_seed = np.random.randint(2 ** 30)


def update_config_for_load_dir(conf, init_conf, load_dir):
    conf.dirs.project_root = proj_conf.project_root
    conf.dirs.imdb_loc = proj_conf.imdb_loc
    conf.dirs.dataset_dir = proj_conf.dataset_dir
    conf.dirs.pose_img_dir = proj_conf.pose_imgs_dir
    conf.dirs.landmarks_file = proj_conf.landmarks_file

    # TODO organise dirs - out files dirs and in files dirs, also remove dir from variable name
    conf.dirs.save_dir = load_dir

    conf.dirs.log_dir = osp.join(conf.dirs.save_dir, init_conf.dirs.log_dir)
    conf.dirs.model_save_dir = osp.join(conf.dirs.save_dir, init_conf.dirs.model_save_dir)
    conf.dirs.result_dir = osp.join(conf.dirs.save_dir, init_conf.dirs.result_dir)
    conf.dirs.sample_dir = osp.join(conf.dirs.save_dir, init_conf.dirs.sample_dir)
    conf.dirs.interpolate_dir = osp.join(conf.dirs.save_dir, init_conf.dirs.interpolate_dir)

    conf.out_files.config_file_loc = osp.join(conf.dirs.save_dir, init_conf.out_files.config_file_loc)
    conf.out_files.config_text_loc = osp.join(conf.dirs.save_dir, init_conf.out_files.config_text_loc)


def update_config_from_args(conf, args, other_args):
    """
    Updates the conf dictionary with variables obtained from args dictionary. Also adds up other args to args if there
    are no errors.

    The config variable corresponding to the args variable is located through the alias dict

    If an alias for the args value passed is not found, then the key is looked up in the conf leaf keys. If a match is
    found then the corresponding variable is updated else the function raises an error.

    :param conf: the config dictionary
    :param args: the args dictionary - commandline arguments
    """

    # Add other args to args
    # Known restrictions to unknown args - No flags

    assert len(other_args) % 2 == 0, "Number of non default arguments should be even"
    other_args_dict = {}
    double_dash = "--"
    num_unk_args = int(len(other_args) / 2)

    for i_key in range(num_unk_args):
        key = other_args[2 * i_key]  # strip away hyphens
        value = other_args[2 * i_key + 1]

        assert key[:2] == double_dash, "Argument key {0} doesn't have a double dash at the beginning".format(key)
        assert double_dash not in value, "Argument value {0} shouldn't have a double dash".format(value)
        other_args_dict[key[2:]] = get_class_value(value)

    # Update args
    args.update(other_args_dict)

    args_dict = {key: val for (key, val) in args.items() if val is not None}

    for arg in args_dict:
        # Find an alias in arg2conf_dict
        conf_var = arg2conf(arg) or locate_in_conf(conf, arg)
        if conf_var:
            # Choose arg only if the key is present in conf
            conf[conf_var[0]][conf_var[1]] = get_class_value(args[arg])
        else:
            msg = "The argument passed '{0}' was not an alias or found in the config dict".format(arg)
            raise Exception(msg)

    conf.args = edict(args_dict)


def get_new_run_name(saved_data_loc, run_name):
    i = 0
    orig_runname = run_name
    while True:
        save_dir = osp.join(saved_data_loc, run_name)
        if osp.exists(save_dir):
            i += 1
            run_name = orig_runname + str(i)
        else:
            break
    return run_name


def get_class_value(value):
    try:
        val_class = float(value)
        if val_class.is_integer():
            val_class = int(val_class)
    except ValueError:
        val_class = value
    return val_class


def update_config(conf, args, other_args):
    update_config_from_args(conf, args, other_args)

    if conf.flags.TEST_BUILD:
        conf.train_param.max_iter = 50
        conf.train_param.model_save_step = 25
        conf.train_param.sample_step = 5
        conf.train_param.log_step = 5
        conf.train_param.lr_update_step = 20
        conf.sample_param.num_samples = 2

    if conf.flags.APPEND_DATETIME:
        conf.params.datetime = datetime.datetime.now().strftime("%d.%m-%H:%M:%S")
        conf.dirs.run_name = '{}-{}'.format(conf.dirs.run_name, conf.params.datetime)
    else:
        # ensure that you get a new run name if a directory already exists
        conf.dirs.run_name = get_new_run_name(conf.dirs.saved_data_loc, conf.dirs.run_name)

    conf.dirs.save_dir = osp.join(conf.dirs.saved_data_loc, conf.dirs.run_name)
    conf.dirs.log_dir = osp.join(conf.dirs.save_dir, conf.dirs.log_dir)
    conf.dirs.model_save_dir = osp.join(conf.dirs.save_dir, conf.dirs.model_save_dir)
    conf.dirs.result_dir = osp.join(conf.dirs.save_dir, conf.dirs.result_dir)
    conf.dirs.sample_dir = osp.join(conf.dirs.save_dir, conf.dirs.sample_dir)
    conf.dirs.interpolate_dir = osp.join(conf.dirs.save_dir, conf.dirs.interpolate_dir)
    conf.dirs.export_code = osp.join(conf.dirs.save_dir, conf.dirs.export_code)

    conf.out_files.config_file_loc = osp.join(conf.dirs.save_dir, conf.out_files.config_file_loc)
    conf.out_files.config_text_loc = osp.join(conf.dirs.save_dir, conf.out_files.config_text_loc)

    ensure_dir_exists(conf.dirs.save_dir, delete_directory=True)
    ensure_dir_exists(conf.dirs.log_dir, delete_directory=True)
    ensure_dir_exists(conf.dirs.model_save_dir, delete_directory=True)
    ensure_dir_exists(conf.dirs.result_dir, delete_directory=True)
    ensure_dir_exists(conf.dirs.sample_dir, delete_directory=True)
    ensure_dir_exists(conf.dirs.interpolate_dir, delete_directory=True)
    # ensure_dir_exists(conf.dirs.export_code, delete_directory=True)


def arg2conf(arg):
    """Converts argument alias to config parameter"""
    arg2conf_dict = arg2conf_alias()
    return arg2conf_dict[arg] if arg in arg2conf_dict else None


def locate_in_conf(conf, arg):
    for category in conf:
        for sub_category in conf[category]:
            if sub_category == arg or sub_category == arg.upper():  # check for FLAGS
                return [category, sub_category]

    return None


### Print and export functions
def export_config(conf):
    save_obj(conf.out_files.config_file_loc, conf)

    ## Text format ##

    with open(conf.out_files.config_text_loc, 'w') as file:
        if bool(conf.args):  # check if args are present
            arg_str = args2str(conf.args)
            file.write(arg_str.__str__())
            file.write('\n')

        cfg_str = cfg2str(conf)
        file.write(cfg_str.__str__())


def pretty_dict(dict_obj, dict_obj_name):
    dict_str = special_string()
    dict_str.add_line(dict_obj_name)
    max_length = max([len(component) for component in dict_obj])
    for component in dict_obj:
        comp_str = component.ljust(max_length + 1)
        sub_comp_str = '\t{0}  :  {1}'.format(comp_str, dict_obj[component])
        dict_str.add_line(sub_comp_str)
    dict_str.newline()
    return dict_str.__str__()


def args2str(args):
    args_str = special_string()
    dict_args_print = {arg: str(args[arg]) for arg in args}
    args_str.append_str(pretty_dict(dict_args_print, 'Arguments passed'))
    return args_str


def cfg2str(conf):
    cfg_str = special_string()
    for category in conf:
        if bool(conf[category]):
            # i.e. if the category is not empty
            category_dict_str = pretty_dict(conf[category], category)
            cfg_str.append_str(category_dict_str)
    return cfg_str
