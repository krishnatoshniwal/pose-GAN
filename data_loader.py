import torch
import numpy as np
import random
from PIL import Image
import os

import torch.utils.data as data
from torchvision import transforms as T

from resources.utils import load_obj
from solver_utils import vis_tensor

# TODO: create imdb so that this doesnt have any randomness
def get_three_nums(max_num):
    lst = list(range(max_num))
    lst = sorted(np.random.choice(lst, 3, replace=False))
    if np.random.rand() > 0.5:
        lst.reverse()
    return lst


def get_spaced_two_random_nums(max_num):
    space = 2
    nums_list = list(range(max_num - space))
    num_random = random.choice(nums_list)
    other_num_random = num_random + space
    mid_num = num_random + int(space / 2)
    if np.random.rand() > 0.5:
        num_random, other_num_random = other_num_random, num_random
    return [num_random, mid_num, other_num_random]


class MultiPieInterpolation(data.Dataset):
    def __init__(self, cfg, selected_poses, transform=None):
        self.img_dir = cfg.dirs.dataset_dir
        self.imdb = load_obj(cfg.dirs.imdb_loc)

        self.transform = transform
        self.offset = 120
        self.selected_poses = selected_poses
        self.landmarks_dict = load_obj(cfg.dirs.landmarks_file)

        self.id_to_class = {id_img: i_class for i_class, id_img in enumerate(self.imdb.keys())}
        self.class_to_id = {i_class: id_img for id_img, i_class in self.id_to_class.items()}

        self.INTERPOLATION_MODE = cfg.train_param.INTERPOLATION_MODE

    def open_and_crop_img(self, filename):
        """ Takes in filename, splits it, open and crop it"""

        # Extract landmarks from dict = sessionID -> personID -> cameraID
        file_attrs = os.path.split(filename)[1]
        file_attrs = file_attrs.split("_")
        session_id = int(file_attrs[1]) - 1
        person_id = int(file_attrs[0])
        camera_id = file_attrs[3]

        landmarks = self.landmarks_dict[session_id][person_id][camera_id][0]

        row_nose_pos = landmarks[2][0]
        col_nose_pos = landmarks[2][1]
        bbox = (col_nose_pos - self.offset, row_nose_pos - self.offset,
                col_nose_pos + self.offset, row_nose_pos + self.offset)

        img = Image.open(os.path.join(self.img_dir, filename))
        img = img.crop(bbox)
        return img

    def get_image_pose(self, id_img, selected_pose, im_id):
        # selected_pose = self.selected_poses[pose_idx]
        img_pose = self.imdb[id_img][selected_pose][im_id]
        img_pose = self.open_and_crop_img(img_pose)
        if self.transform:
            img_pose = self.transform(img_pose)
        else:
            img_pose = np.array(img_pose)
        return np.array(selected_pose), img_pose

    def __getitem__(self, class_index):
        id_img = self.class_to_id[class_index]
        id_pose_images = self.imdb[id_img]
        im_id = random.choice(list(id_pose_images[0].keys()))

        if self.INTERPOLATION_MODE:
            # chosen_idx = get_spaced_two_random_nums(len(self.selected_poses))
            chosen_idx = get_three_nums(len(self.selected_poses))
            pose_0, img_pose_0 = self.get_image_pose(id_img, self.selected_poses[chosen_idx[0]], im_id)
            pose_1, img_pose_1 = self.get_image_pose(id_img, self.selected_poses[chosen_idx[1]], im_id)
            pose_2, img_pose_2 = self.get_image_pose(id_img, self.selected_poses[chosen_idx[2]], im_id)

        else:
            chosen_poses = np.random.choice(np.array(self.selected_poses), 2, replace=True)
            pose_0, img_pose_0 = self.get_image_pose(id_img, chosen_poses[0], im_id)
            pose_1, img_pose_1 = self.get_image_pose(id_img, chosen_poses[1], im_id)
            pose_2, img_pose_2 = 0, 0

        class_index = np.array(class_index)

        return class_index, img_pose_0, pose_0, img_pose_1, pose_1, img_pose_2, pose_2

    def get_sample_imgs(self, available_poses, num_peeps, device):
        def get_img(p_id, pose, im_id):
            img_pose = self.open_and_crop_img(self.imdb[p_id][pose][im_id])
            if self.transform is None:
                img_pose = np.array(img_pose)
            else:
                img_pose = self.transform(img_pose)
            return img_pose

        p_ids = [self.class_to_id[i] for i in range(num_peeps)]

        imgs1, imgs2 = [], []
        sample_poses1, sample_poses2 = [], []
        for p_id in p_ids:
            sample_pose1 = random.choice(available_poses)
            sample_pose2 = random.choice(available_poses)

            im_id = random.choice(list(self.imdb[p_id][sample_pose1].keys()))

            imgs1.append(get_img(p_id, sample_pose1, im_id))
            imgs2.append(get_img(p_id, sample_pose2, im_id))
            sample_poses1.append(sample_pose1)
            sample_poses2.append(sample_pose2)

        stack_dev = lambda t: torch.stack(t).to(device)
        return stack_dev(imgs1), sample_poses1, stack_dev(imgs2), sample_poses2

    def get_interp_imgs(self, interp_pose1, interp_pose2, num_peeps, device):
        def get_img(p_id, pose, im_id):
            img_pose = self.open_and_crop_img(self.imdb[p_id][pose][im_id])
            if self.transform is None:
                img_pose = np.array(img_pose)
            else:
                img_pose = self.transform(img_pose)
            return img_pose

        p_ids = [self.class_to_id[i] for i in range(num_peeps)]

        imgs1, imgs2 = [], []
        for p_id in p_ids:
            im_id = random.choice(list(self.imdb[p_id][interp_pose1].keys()))

            imgs1.append(get_img(p_id, interp_pose1, im_id))
            imgs2.append(get_img(p_id, interp_pose2, im_id))

        return torch.stack(imgs1).to(device), torch.stack(imgs2).to(device)

    def __len__(self):
        return len(self.id_to_class)


def get_loader(cfg):
    # transforms.append(T.CenterCrop(args['crop_size']))
    img_size = cfg.model_param.img_size

    num_workers = cfg.train_param.num_workers
    batch_size = cfg.train_param.batch_size

    transforms = [
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]

    transforms = T.Compose(transforms)

    dataset = MultiPieInterpolation(cfg, cfg.train_param.poses, transform=transforms)
    data_loader = data.DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size, drop_last=True)
    num_classes = len(dataset)
    return data_loader, num_classes
