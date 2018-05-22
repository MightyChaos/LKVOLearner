# from torchvision.transforms import Scale # it appears torchvision has some bugs, work on that latter
import torch.nn as nn
from torch.nn import AvgPool2d
from torch.nn.functional import conv2d
from torch.autograd import Variable
from timeit import default_timer as timer
import torch
import numpy as np

import matplotlib.pyplot as plt
import scipy.io as sio
import os



class ImageSmoothLayer(nn.Module):
    def __init__(self, chan):
        super(ImageSmoothLayer, self).__init__()
        F = torch.FloatTensor([[0.0751,   0.1238,    0.0751],
                              [0.1238,   0.2042,    0.1238],
                              [0.0751,   0.1238,    0.0751]]).view(1, 1, 3, 3)
        self.register_buffer('smooth_kernel', F)
        if chan>1:
            f = F
            F = torch.zeros(chan, chan, 3, 3)
            for i in range(chan):
                F[i, i, :, :] = f
        self.register_buffer('smooth_kernel_K', F)
        self.reflection_pad_func = torch.nn.ReflectionPad2d(1)

    def forward(self, input):
        output_dim = input.dim()
        output_size = input.size()
        if output_dim==2:
            F = self.smooth_kernel
            input = input.unsqueeze(0).unsqueeze(0)
        elif output_dim==3:
            F = self.smooth_kernel
            input = input.unsqueeze(1)
        else:
            F = self.smooth_kernel_K

        x = self.reflection_pad_func(input)

        x = conv2d(input=x,
                    weight=Variable(F),
                    stride=1,
                    padding=0)

        if output_dim==2:
            x =  x.squeeze(0).squeeze(0)
        elif output_dim==3:
            x =  x.squeeze(1)

        return x


class ImagePyramidLayer(nn.Module):
    def __init__(self, chan, pyramid_layer_num):
        super(ImagePyramidLayer, self).__init__()
        self.pyramid_layer_num = pyramid_layer_num
        F = torch.FloatTensor([[0.0751,   0.1238,    0.0751],
                              [0.1238,   0.2042,    0.1238],
                              [0.0751,   0.1238,    0.0751]]).view(1, 1, 3, 3)
        self.register_buffer('smooth_kernel', F)
        if chan>1:
            f = F
            F = torch.zeros(chan, chan, 3, 3)
            for i in range(chan):
                F[i, i, :, :] = f
        self.register_buffer('smooth_kernel_K', F)
        self.avg_pool_func = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.reflection_pad_func = torch.nn.ReflectionPad2d(1)


    def downsample(self, input):
        output_dim = input.dim()
        output_size = input.size()
        if output_dim==2:
            F = self.smooth_kernel
            input = input.unsqueeze(0).unsqueeze(0)
        elif output_dim==3:
            F = self.smooth_kernel
            input = input.unsqueeze(1)
        else:
            F = self.smooth_kernel_K

        x = self.reflection_pad_func(input)

        x = conv2d(input=x,
                    weight=Variable(F),
                    stride=1,
                    padding=0)
        # remove here if not work out
        padding = [0, int(np.mod(input.size(-1), 2)), 0, int(np.mod(input.size(-2), 2))]
        x = torch.nn.ReplicationPad2d(padding)(x)
        # -----
        x = self.avg_pool_func(x)

        if output_dim==2:
            x =  x.squeeze(0).squeeze(0)
        elif output_dim==3:
            x =  x.squeeze(1)

        return x


    def forward(self, input, do_detach=True):
        pyramid = [input]
        for i in range(self.pyramid_layer_num-1):
            img_d = self.downsample(pyramid[i])
            if isinstance(img_d, Variable) and do_detach:
                img_d = img_d.detach()
            pyramid.append(img_d)
            assert(np.ceil(pyramid[i].size(-1)/2) == img_d.size(-1))
        return pyramid


    def get_coords(self, imH, imW):
        x_pyramid = [np.arange(imW)+.5]
        y_pyramid = [np.arange(imH)+.5]
        for i in range(self.pyramid_layer_num-1):
            offset = 2**i
            stride = 2**(i+1)
            x_pyramid.append(np.arange(offset, offset + stride*np.ceil(x_pyramid[i].shape[0]/2), stride))
            y_pyramid.append(np.arange(offset, offset + stride*np.ceil(y_pyramid[i].shape[0]/2), stride))

        return x_pyramid, y_pyramid



if __name__ == "__main__":
    imH = 128
    imW = 416
    img = Variable(torch.randn(3, 128, 416))
    n = ImagePyramidLayer(3, 7)
    x_pyramid, y_pyramid = n.get_coords(imH, imW)
    t = timer()
    # pyramid, x_pyramid, y_pyramid = buildImagePyramid(Variable(torch.rand(1, 3, 256, 320)), 6)
    pyramid = n.forward(img)
    print(timer()-t)

    for i in range(n.pyramid_layer_num):
        print(x_pyramid[i].shape[0])
        print(y_pyramid[i].shape[0])
        print(pyramid[i].size())
    # print(pyramid)
