import numpy as np
import torch.utils.data as data
import random
from PIL import Image
import glob
import torch
import os
from collections import defaultdict
from torchvision import transforms as T
import sys

# TODO clean import
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from resources.utils import save_obj
import proj_conf
import config


class ImdbBuilder():
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.camera_id2pose = {'110': 0, '120': 15, '090': 30, '080': 45, '130': 60,
                         '140': 75, '051': 90, '050': 105, '041': 120, '190': 135,
                         '200': 150, '010': 165, '240': 180}

        self.faulty_ids = [16, 31, 37, 55, 61, 63, 115, 118, 126, 129, 153,
                           167, 174, 175, 184, 209, 210, 238, 239, 257, 280, 330]
        print("Not training on {} IDs: ".format(len(self.faulty_ids)), self.faulty_ids)

    def build_imdb(self):
        imdb = defaultdict(dict)

        cwd = os.getcwd()
        print("Building MultiPIE dataset")

        for i in range(4):
            multiview_dir = os.path.join(self.dataset_dir, 'session0{}'.format(i + 1), 'multiview')
            os.chdir(multiview_dir)

            for filename in glob.glob("*/*/*/*.png"):
                file = filename.split("/")[-1].rstrip('.png')
                file_ids = file.split("_")
                person_id = int(file_ids[0])
                session_id = file_ids[1]
                recording_id = file_ids[2]
                camera_id = file_ids[3]
                image_id = file_ids[4]

                im_id = '_'.join([session_id, recording_id, image_id])
                # Ignore cameras placed at a height
                if camera_id not in self.camera_id2pose.keys():
                    continue
                if person_id in self.faulty_ids:
                    continue

                pose = self.camera_id2pose[camera_id]

                if pose not in imdb[person_id]:
                    imdb[person_id][pose] = {}
                if im_id in imdb[person_id][pose]:
                    print('image already present')
                imdb[person_id][pose][im_id] = os.path.join(multiview_dir, filename)
            os.chdir(cwd)

        for key in self.faulty_ids:
            assert key not in imdb.keys()

        print("Built imdb on {} IDs.....".format(len(imdb.keys())))
        self.imdb = imdb
        return imdb


if __name__ == '__main__':
    imdb_obj = ImdbBuilder(proj_conf.dataset_dir)
    imdb = imdb_obj.build_imdb()
    save_obj(proj_conf.imdb_loc, imdb)
