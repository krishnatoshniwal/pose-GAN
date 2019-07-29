import numpy as np
import os
import os.path as osp
import yaml
from math import sqrt, atan2, pi
# from scipy.misc import imread
import pickle
from proj_conf import project_root, data_loc
from resources.utils import load_obj, save_obj


def add_root(dir_, root=project_root):
    return osp.join(root, dir_)


def rpy(rmat):
    '''
    Assumes that the rotation_mat is a list or array of shape 3x3
    Returns the array [phi,theta,psi]
    '''
    r11 = rmat[0][0]
    r21 = rmat[1][0]
    r31 = rmat[2][0]
    r32 = rmat[2][1]
    r33 = rmat[2][2]

    rad2deg = 180.0 / pi

    phi = atan2(r32, r33) * rad2deg
    theta = atan2(-r31, sqrt(r32 ** 2 + r33 ** 2)) * rad2deg
    psi = atan2(r21, r11) * rad2deg

    return (phi, theta, psi)


def get_imdb_for_dir(dir_loc):
    """
    Parses the yaml file for the tless dataset
    """
    # dir_file = os.path.dirname(file_loc)
    gt_yaml_loc = add_root('gt.yml', dir_loc)
    info_yaml_loc = add_root('info.yml', dir_loc)
    with open(gt_yaml_loc, 'r') as file_ptr:
        gt_yaml = yaml.load(file_ptr)

    with open(info_yaml_loc, 'r') as file_ptr:
        info_yaml = yaml.load(file_ptr)

    imdb = {0: [], 1: []}
    # process the information loaded
    print('Current dir {}'.format(dir_loc))
    for img_key in gt_yaml.keys():
        image_loc = 'rgb/%04d.jpg' % (img_key)

        img_data = gt_yaml[img_key][0]
        mode = info_yaml[img_key]['mode']
        elev = info_yaml[img_key]['elev']

        rotation_list = img_data['cam_R_m2c']

        rotation_matrix = [rotation_list[i::3] for i in range(3)]
        translation_matrix = img_data['cam_t_m2c']
        obj_id = img_data['obj_id']
        obj_bb = img_data['obj_bb']

        data_pt = {}
        data_pt['loc'] = add_root(image_loc, dir_loc)
        data_pt['rpy'] = rpy(rotation_matrix)
        data_pt['category'] = obj_id
        data_pt['id'] = img_key
        data_pt['bb'] = obj_bb
        # im = imread(data_pt['loc'])
        imdb[mode].append(data_pt)

    return imdb


def _build_split(loc):
    # sweep the folder for object categories
    object_cat_dirs = os.listdir(loc)
    # object_ids = list(map(int, object_cat_dirs))

    imdb = {}
    for object_cat_dir in object_cat_dirs:
        # for object_cat_dir in ['01']:
        imdb_cat = get_imdb_for_dir(add_root(object_cat_dir, loc))
        imdb[int(object_cat_dir)] = imdb_cat
    return imdb


def main():
    # source
    t_less_folder = add_root(data_loc)
    train_folder = t_less_folder + 'train_canon/'
    test_folder = t_less_folder + 'test_canon/'
    imdb_train_loc = add_root('imdb_train.pkl')
    imdb_test_loc = add_root('imdb_test.pkl')

    imdb_train = _build_split(train_folder)
    imdb_test = _build_split(test_folder)

    save_obj(imdb_train, imdb_train_loc)
    save_obj(imdb_test, imdb_test_loc)


if __name__ == '__main__':
    main()
