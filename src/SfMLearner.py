from DirectVOLayer import DirectVO
from networks import VggDepthEstimator, PoseNet, PoseExpNet
from ImagePyramid import ImagePyramidLayer
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

import itertools

from timeit import default_timer as timer

class FlipLR(nn.Module):
    def __init__(self, imW, dim_w):
        super(FlipLR, self).__init__()
        inv_indices = torch.arange(imW-1, -1, -1).long()
        self.register_buffer('inv_indices', inv_indices)
        self.dim_w = dim_w


    def forward(self, input):
        return input.index_select(self.dim_w, Variable(self.inv_indices))



class SfMLearner(nn.Module):
    def __init__(self, img_size=[128, 416], ref_frame_idx=1,
        lambda_S=.5, lambda_E=0.01, use_ssim=True, smooth_term = 'lap',
        use_expl_mask=False, gpu_ids=[0]):
        super(SfMLearner, self).__init__()
        self.sfmkernel = nn.DataParallel(SfMKernel(img_size, smooth_term = smooth_term, use_expl_mask=use_expl_mask),
                            device_ids=gpu_ids)
        self.ref_frame_idx = ref_frame_idx
        self.lambda_S = lambda_S
        self.lambda_E = lambda_E
        self.use_ssim = use_ssim
        self.use_expl_mask = use_expl_mask

    def forward(self, frames, camparams, max_lk_iter_num=10):
        cost, photometric_cost, smoothness_cost, ref_frame, ref_inv_depth, ref_expl_mask \
            = self.sfmkernel.forward(frames, camparams, self.ref_frame_idx,
            self.lambda_S, self.lambda_E, use_ssim=self.use_ssim)
        return cost.mean(), photometric_cost.mean(), smoothness_cost.mean(), ref_frame, ref_inv_depth, ref_expl_mask

    def save_model(self, file_path):
        torch.save(self.cpu().sfmkernel.module.depth_net.state_dict(),
            file_path+'_depth_net.pth')
        torch.save(self.sfmkernel.module.pose_net.state_dict(),
            file_path+'_pose_net.pth')
        self.cuda()

    def load_model(self, file_path):
        self.sfmkernel.module.depth_net.load_state_dict(torch.load(file_path+'_depth_net.pth'))
        self.sfmkernel.module.pose_net.load_state_dict(torch.load(file_path+'_pose_net.pth'))

    def init_weights(self):
        self.sfmkernel.module.depth_net.init_weights()

    def get_parameters(self):
        return itertools.chain(self.sfmkernel.module.depth_net.parameters(),
                    self.sfmkernel.module.pose_net.parameters())



class SfMKernel(nn.Module):
    """
     only support single training isinstance
    """
    def __init__(self, img_size=[128, 416], smooth_term = 'lap', use_expl_mask=False):
        super(SfMKernel, self).__init__()
        self.img_size = img_size
        self.fliplr_func = FlipLR(imW=img_size[1], dim_w=3)
        self.vo = DirectVO(imH=img_size[0], imW=img_size[1], pyramid_layer_num=4)
        self.depth_net = VggDepthEstimator(img_size)
        if use_expl_mask:
            self.pose_net = PoseExpNet(3)
        else:
            self.pose_net = PoseNet(3)
        self.pyramid_func = ImagePyramidLayer(chan=1, pyramid_layer_num=4)
        self.smooth_term = smooth_term
        self.use_expl_mask = use_expl_mask


    def forward(self, frames, camparams, ref_frame_idx, lambda_S=.5, lambda_E=.01, do_data_augment=True, use_ssim=True):
        assert(frames.size(0) == 1 and frames.dim() == 5)
        frames = frames.squeeze(0)
        camparams = camparams.squeeze(0).data


        if do_data_augment:
            if np.random.rand()>.5:
                frames = self.fliplr_func(frames)
                camparams[2] = self.img_size[1] - camparams[2]

        bundle_size = frames.size(0)
        src_frame_idx = tuple(range(0,ref_frame_idx)) + tuple(range(ref_frame_idx+1,bundle_size))
        frames_pyramid = self.vo.pyramid_func(frames)
        ref_frame_pyramid = [frame[ref_frame_idx, :, :, :] for frame in frames_pyramid]
        src_frames_pyramid = [frame[src_frame_idx, :, :, :] for frame in frames_pyramid]


        self.vo.setCamera(fx=camparams[0], cx=camparams[2],
                            fy=camparams[4], cy=camparams[5])
        self.vo.init_xy_pyramid(ref_frame_pyramid)
        if self.use_expl_mask:
            p, expl_mask_pyramid = self.pose_net.forward((frames.view(1, -1, frames.size(2), frames.size(3))-127) / 127)
            expl_mask_reg_cost = 0
            for mask in expl_mask_pyramid:
                expl_mask_reg_cost += mask.mean()
            ref_expl_mask_pyramid = [mask.squeeze(0)[ref_frame_idx, ...] for mask in expl_mask_pyramid]
            src_expl_mask_pyramid = [mask.squeeze(0)[src_frame_idx, ...] for mask in expl_mask_pyramid]
            expl_mask = ref_expl_mask_pyramid[0]

        else:
            p = self.pose_net.forward((frames.view(1, -1, frames.size(2), frames.size(3))-127) / 127)
            ref_expl_mask_pyramid = None
            src_expl_mask_pyramid = None
            expl_mask_reg_cost = 0
            expl_mask = None

        rot_mat_batch = self.vo.twist2mat_batch_func(p[0,:,0:3])
        trans_batch = p[0,:,3:6]

        inv_depth_pyramid = self.depth_net.forward((frames-127)/127)
        inv_depth_mean_ten = inv_depth_pyramid[0].mean()*0.1 #uncommment this to use normalization

        # normalize
        #trans_batch = trans_batch*inv_depth_mean_ten
        inv_depth_norm_pyramid = [depth/inv_depth_mean_ten for depth in inv_depth_pyramid]

        ref_inv_depth_pyramid = [depth[ref_frame_idx, :, :] for depth in inv_depth_norm_pyramid]
        src_inv_depth_pyramid = [depth[src_frame_idx, :, :] for depth in inv_depth_norm_pyramid]

        photometric_cost = self.vo.compute_phtometric_loss(
                                                ref_frame_pyramid,
                                                src_frames_pyramid,
                                                ref_inv_depth_pyramid,
                                                src_inv_depth_pyramid,
                                                rot_mat_batch, trans_batch,
                                                levels=[0,1,2,3], use_ssim=use_ssim,
                                                ref_expl_mask_pyramid=ref_expl_mask_pyramid,
                                                src_expl_mask_pyramid=src_expl_mask_pyramid)
        # compute smoothness smoothness loss
        # instead of directly compute the loss on the finest level, it's evaluated on the downsamples.
        inv_depth0_pyramid = self.pyramid_func(inv_depth_norm_pyramid[0], do_detach=False)
        smoothness_cost = self.vo.multi_scale_image_aware_smoothness_cost(inv_depth0_pyramid, frames_pyramid, levels=[2,3], type=self.smooth_term) \
                            + self.vo.multi_scale_image_aware_smoothness_cost(inv_depth_norm_pyramid, frames_pyramid, levels=[2,3], type=self.smooth_term)

        cost = photometric_cost + lambda_S*smoothness_cost - lambda_E*expl_mask_reg_cost
        
        return cost, photometric_cost, smoothness_cost, ref_frame_pyramid[0], ref_inv_depth_pyramid[0]*inv_depth_mean_ten, expl_mask
