import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import config
import numpy as np
import torch
import os.path as osp
from resources.utils import load_obj


def set_seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)


def load_config(config_file_loc):
    # the directory must also contain the model checkpoint for later loading
    config_file = osp.join(config_file_loc, 'config.pkl')
    cfg = load_obj(config_file)
    cfg.flags.LOADED_CONFIG = True
    return cfg


def main(cfg):
    # For fast training.
    cudnn.benchmark = True
    solver = Solver(cfg)
    #
    # if config.mode == 'train':
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--run_name', type=str, help='Name of the current run')
    parser.add_argument('--c_dim', type=int, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--crop_size', type=int, help='Center Crop')
    parser.add_argument('--img_size', type=int, help='image resolution')
    parser.add_argument('--d_conv_dim', type=int, help='number of conv filters in the first layer of D')
    # parser.add_argument('--g_repeat_num', type=int, help='number of residual blocks in G')
    parser.add_argument('--lambda_gp', type=float, help='weight for gradient penalty')
    parser.add_argument('--lambda_lcnn', type=float, help='weight for LightCNN')
    parser.add_argument('--lambda_cls', type=float, help='weight for classification loss')
    parser.add_argument('--lambda_tvr', type=float, help='weight for total variation loss')
    parser.add_argument('--lambda_l1', type=float, help='weight for l1 loss')
    parser.add_argument('--lambda_cyc', type=float, help='weight for cyclic loss')
    parser.add_argument('--cuda', type=int, help='Train on device id')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, help='mini-batch size')
    parser.add_argument('--max_iters', type=int, help='number of total iterations for training D')
    parser.add_argument('--g_lr', type=float, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, help='beta2 for Adam optimizer')
    parser.add_argument('--load_dir', type=str, help='The saved directory of the model')
    parser.add_argument('--use_tensorboard', type=bool)
    parser.add_argument('--poses', type=list, help='Selected poses')
    parser.add_argument('--test_build', action='store_true',
                        help='Test the current build with very small number of iterations')

    parser.add_argument('--sample_step', type=int)
    # Test configuration.
    # parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    # parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    args, other_args = parser.parse_known_args()
    args = vars(args)  # Convert to dict

    init_cfg = config.cfg
    if args['load_dir']:
        cfg = load_config(args['load_dir'])
        config.update_config_from_args(cfg, args, other_args)
        config.update_config_for_load_dir(cfg, init_cfg, args['load_dir'])

    else:
        config.initialise_config(init_cfg)  # Initialise some basic variables
        cfg = init_cfg
        set_seed(cfg.params.rand_seed)

        config.update_config(cfg, args, other_args)
        config.export_config(cfg)
        config.export_files(cfg)
    dev_id = cfg.params.cuda_device_id
    cfg.params.cuda_device_id = 'cuda:' + str(dev_id) if type(dev_id) is int else dev_id

    print(config.cfg2str(cfg))

    main(cfg)
