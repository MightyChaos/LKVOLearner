from __future__ import division
import os
import math
import scipy.misc
import numpy as np
from glob import glob
from pose_evaluation_utils import dump_pose_seq_TUM, mat2euler

from DirectVOLayer import DirectVO
from networks_old import VggDepthEstimator
import torch.nn as nn
from torch.autograd import Variable
import torch
from ImagePyramid import ImagePyramidLayer

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--img_height", type=int, default=128, help="Image height")
parser.add_argument("--img_width", type=int, default=416, help="Image width")
parser.add_argument("--seq_length", type=int, default=5, help="Sequence length for each example")
parser.add_argument("--test_seq", type=int, default=9, help="Sequence id to test")
parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
parser.add_argument("--depth_file", type=str, default=None, help="depth map file")
FLAGS = parser.parse_args()

def load_image_sequence(dataset_dir,
                        frames,
                        tgt_idx,
                        seq_length,
                        img_height,
                        img_width):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        curr_img = scipy.misc.imread(img_file)
        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        if o == -half_offset:
            image_seq = curr_img
        else:
            image_seq = np.hstack((image_seq, curr_img))
    return image_seq

def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    # TODO: unnecessary to check if the drives match
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False

def main():
    vo = DirectVO(imH=FLAGS.img_height, imW=FLAGS.img_width, pyramid_layer_num=5, max_itr_num=100)
    vo.setCamera(fx=292.4844, cx=206.2069,
                        fy=242.9359, cy=62.4305)
    vo.cuda()
    pyramid_func = ImagePyramidLayer(chan=3,
                            pyramid_layer_num=5)
    pyramid_func.cuda()

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    seq_dir = os.path.join(FLAGS.dataset_dir, 'sequences', '%.2d' % FLAGS.test_seq)
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]
    with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])
    max_src_offset = (FLAGS.seq_length - 1)//2

    pred_depths_np = np.load(FLAGS.depth_file)
    pred_depths = []
    for tgt_idx in range(N):
    # for tgt_idx in range(20):
        if not is_valid_sample(test_frames, tgt_idx, FLAGS.seq_length):
            continue
        if tgt_idx % 100 == 0:
            print('Progress: %d/%d' % (tgt_idx, N))
        # TODO: currently assuming batch_size = 1
        image_seq = load_image_sequence(FLAGS.dataset_dir,
                                        test_frames,
                                        tgt_idx,
                                        FLAGS.seq_length,
                                        FLAGS.img_height,
                                        FLAGS.img_width)
        frames_np_list = [image_seq[:,0:FLAGS.img_width,:]]
        for i in range(FLAGS.seq_length-1):
            frames_np_list.append(image_seq[:,FLAGS.img_width*(i+1):FLAGS.img_width*(i+2),:])
        frames_np = np.asarray(frames_np_list)
        frames = Variable(torch.from_numpy(frames_np).cuda().float(), volatile=False)
        frames = frames.permute(0, 3, 1, 2).contiguous()
        # frames = frames.view(FLAGS.img_height, FLAGS.img_width, FLAGS.seq_length, 3).permute(2,3,0,1).contiguous()
        print(frames.size())
        bundle_size = frames.size(0)
        ref_frame_idx = 2
        src_frame_idx = tuple(range(0,ref_frame_idx)) + tuple(range(ref_frame_idx+1,bundle_size))
        ref_frame = frames[ref_frame_idx, :, :, :]
        src_frames = frames[src_frame_idx, :, :, :]
        print(ref_frame.size())
        # _, inv_depth0_pyramid = depth_net.forward((ref_frame.unsqueeze(0)-127)/127)
        # ref_inv_depth0_pyramid = [depth[0, :, :] for depth in inv_depth0_pyramid]
        ref_inv_depth = Variable(torch.from_numpy(pred_depths_np[tgt_idx, :, :]).cuda())
        ref_inv_depth0_pyramid = pyramid_func(1/ref_inv_depth)
        ref_frame_pyramid = pyramid_func(ref_frame)
        vo.init(ref_frame_pyramid=ref_frame_pyramid, inv_depth_pyramid=ref_inv_depth0_pyramid)
        pred_poses = np.zeros([bundle_size-1, 6])

        rot_mat = Variable(torch.eye(3).view(1,3,3).cuda())
        trans = Variable(torch.zeros(1,3).cuda())
        print((bundle_size-1)/2)
        for i in range(int((bundle_size-1)/2)):
            print("-----------------")
            src_id = int((bundle_size-1)/2-1-i)
            rot_mat, trans, _ = vo.update2(src_frames[src_id:src_id+1,:,:,:], rot_mat, trans)
            rot_mat_np = rot_mat.data.cpu().numpy()
            trans_np = trans.data.cpu().numpy()
            pred_poses[src_id,0:3] = trans_np[0,:]
            az, ay, ax = mat2euler(rot_mat_np[0,:,:])
            pred_poses[src_id,3:6] = np.asarray([ax, ay, az])

        rot_mat = Variable(torch.eye(3).view(1,3,3).cuda())
        trans = Variable(torch.zeros(1,3).cuda())
        for i in range(int((bundle_size-1)/2)):
            print("-----------------")
            src_id = int((bundle_size-1)/2+i)
            rot_mat, trans, _ = vo.update2(src_frames[src_id:src_id+1,:,:,:], rot_mat, trans)
            rot_mat_np = rot_mat.data.cpu().numpy()
            trans_np = trans.data.cpu().numpy()
            pred_poses[src_id,0:3] = trans_np[0,:]
            az, ay, ax = mat2euler(rot_mat_np[0,:,:])
            pred_poses[src_id,3:6] = np.asarray([ax, ay, az])

        # rot_mat_batch, trans_batch, ref_frame_pyramid, src_frames_pyramid = \
        #     vo.forward(ref_frame, src_frames, ref_inv_depth0_pyramid)
        #
        # rot_mat_batch_np = rot_mat_batch.data.cpu().numpy()
        # trans_batch_np = trans_batch.data.cpu().numpy()
        # pred_poses = np.zeros([bundle_size-1, 6])
        # for i in range(bundle_size-1):
        #     pred_poses[i,0:3] = trans_batch_np[i,:]
        #     az, ay, ax = mat2euler(rot_mat_batch_np[i,:,:])
        #     pred_poses[i,3:6] = np.asarray([ax, ay, az])

        pred_depths.append(ref_inv_depth0_pyramid[0].data.cpu().numpy())
        # Insert the target pose [0, 0, 0, 0, 0, 0]
        pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=0)
        curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
        out_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
        print(pred_poses)
        dump_pose_seq_TUM(out_file, pred_poses, curr_times)
    import scipy.io as sio
    sio.savemat(os.path.join(FLAGS.output_dir, 'pred_depths_%02d' % (FLAGS.test_seq)), {'D': np.asarray(pred_depths)})

main()
