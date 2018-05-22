from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
from PIL import Image
import os


class KCSdataset(Dataset):
    """KITTIdataset"""
    def __init__(self, data_root_path='/home/chaoyang/LKVOLearner', kitti_list_file='train.txt', kitti_data_root_path='data_kitti',
                       cs_list_file='train.txt', cs_data_root_path='data_cs', img_size=[128, 416], bundle_size=3):
        self.data_root_path = data_root_path
        self.img_size = img_size
        self.bundle_size = bundle_size
        self.frame_pathes = []
        # load kitti dataset
        kitti_list_file = os.path.join(data_root_path, kitti_data_root_path, kitti_list_file)
        with open(kitti_list_file) as file:
            for line in file:
                frame_path = line.strip()
                seq_path, frame_name = frame_path.split(" ")
                frame_path = os.path.join(kitti_data_root_path, seq_path, frame_name)
                self.frame_pathes.append(frame_path)
                print(frame_path)
        # load cityscape dataset

        # cs_list_file = os.path.join(data_root_path, cs_data_root_path, cs_list_file)
        # with open(cs_list_file) as file:
        #     for line in file:
        #         frame_path = line.strip()
        #         seq_path, frame_name = frame_path.split(" ")
        #         frame_path = os.path.join(cs_data_root_path, seq_path, frame_name)
        #         self.frame_pathes.append(frame_path)
        #         print(frame_path)

    def __len__(self):
        return len(self.frame_pathes)

    def __getitem__(self, item):
        # read camera intrinsics
        cam_file = os.path.join(self.data_root_path, self.frame_pathes[item]+'_cam.txt')
        with open(cam_file) as file:
            cam_intrinsics = [float(x) for x in next(file).split(',')]
        camparams = np.asarray(cam_intrinsics)

        # read image bundle
        img_file = os.path.join(self.data_root_path, self.frame_pathes[item]+'.jpg')
        frames_cat = np.array(Image.open(img_file))
        # slice the image into #bundle_size number of images
        frame_list = []
        for i in range(self.bundle_size):
            frame_list.append(frames_cat[:,i*self.img_size[1]:(i+1)*self.img_size[1],:])
        frames = np.asarray(frame_list).astype(float).transpose(0, 3, 1, 2)


        return frames, camparams
