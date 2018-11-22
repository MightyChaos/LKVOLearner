from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
from PIL import Image
import os


class KITTIdataset(Dataset):
    """KITTIdataset"""
    def __init__(self, list_file='train.txt', data_root_path='/home/chaoyang/LKVOLearner/data_kitti', img_size=[128, 416], bundle_size=3):
        self.data_root_path = data_root_path
        self.img_size = img_size
        self.bundle_size = bundle_size
        self.frame_pathes = []
        list_file = os.path.join(data_root_path, list_file)
        with open(list_file) as file:
            for line in file:
                frame_path = line.strip()
                seq_path, frame_name = frame_path.split(" ")
                # print(seq_path)
                if seq_path in ['2011_09_26_drive_0119_sync_02', '2011_09_28_drive_0225_sync_02',
                                '2011_09_29_drive_0108_sync_02', '2011_09_30_drive_0072_sync_02',
                                '2011_10_03_drive_0058_sync_02', '2011_09_29_drive_0108_sync_03']:
                    print(seq_path)
                    continue
                frame_path = os.path.join(seq_path, frame_name)
                self.frame_pathes.append(frame_path)
                # print(frame_path)
        # self.frame_pathes = self.frame_pathes[0:40000:800]

    def __len__(self):
        return len(self.frame_pathes)

    def __getitem__(self, item):
        # read camera intrinsics
        cam_file = os.path.join(self.data_root_path, self.frame_pathes[item]+'_cam.txt')
        with open(cam_file) as file:
            cam_intrinsics = [float(x) for x in next(file).split(',')]
        # camparams = dict(fx=cam_intrinsics[0], cx=cam_intrinsics[2],
        #             fy=cam_intrinsics[4], cy=cam_intrinsics[5])
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

if __name__ == "__main__":
    dataset = KITTIdataset()
    dataset.__getitem__(0)
    for i, data in enumerate(dataset):
        print(data[1])
