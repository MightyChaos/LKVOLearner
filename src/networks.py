import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np


DISP_SCALING = 10
MIN_DISP = 0.01

def normalize_convfeat(convfeat):
    max_val = convfeat.max()
    min_val = convfeat.min()
    return convfeat/((max_val-min_val)*.5 + 1e-10)

def normalize_convfeat_pyramid(input_pyramid):
    return [normalize_convfeat(x) for x in input_pyramid]

class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size):
        super(ConvBlock, self).__init__()
        p = int(np.floor((kernel_size - 1) / 2))
        self.activation_fn = nn.ELU()
        self.conv1 = Conv(input_nc, output_nc, kernel_size, 1, p, self.activation_fn)
        # self.conv2 = Conv(output_nc, output_nc, kernel_size, 2, p)
        self.conv2 = Conv(output_nc, output_nc, kernel_size, 1, p, None)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        padding = [0, int(np.mod(input.size(-1), 2)), 0, int(np.mod(input.size(-2), 2))]
        x_pad = torch.nn.ReplicationPad2d(padding)(x)
        return torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)(self.activation_fn(x_pad))


class UpConv(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, scale, output_padding):
        super(UpConv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=input_nc,
                                                out_channels=output_nc,
                                                 kernel_size=2,
                                                 bias=True,
                                                 stride=2,
                                                 padding=0)
        self.activation_fn = nn.ELU()
    def forward(self, input):
        return self.activation_fn(self.deconv(input))

class Conv(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, activation_func=nn.ELU()):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_nc,
                              out_channels=output_nc,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0,
                              bias=True)
        self.activation_fn = activation_func
        self.pad_fn = nn.ReplicationPad2d(padding)

    def forward(self, input):
        if self.activation_fn == None:
            return self.conv(self.pad_fn(input))
        else:
            return self.activation_fn(self.conv(self.pad_fn(input)))

class InvDepth(nn.Module):
    def __init__(self, input_nc):
        super(InvDepth, self).__init__()
        self.conv = Conv(input_nc, 1, 3, 1, 1, None)
        self.batch_norm = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.sigmoid(self.batch_norm(self.conv(input)))*DISP_SCALING + MIN_DISP


def compute_conv_output_size(input_size, kernel_size, stride, padding):
    output_size = [0, 0]
    for i in range(len(input_size)):
        output_size[i] = int(np.floor((input_size[i] + 2*padding - kernel_size)/stride) + 1)
    return output_size

class HalfSigmoid(nn.Module):
    def __init__(self):
        super(HalfSigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return torch.max(input, self.sigmoid(input))


class VggDepthEstimator(nn.Module):
    def __init__(self, input_size):
        super(VggDepthEstimator, self).__init__()
        self.conv_layers = nn.ModuleList([ConvBlock(3, 32, 7)])
        conv_feat_sizes = [compute_conv_output_size(input_size, 7, 2, 3)]

        self.conv_layers.append(ConvBlock(32, 64, 5))
        conv_feat_sizes.append(compute_conv_output_size(conv_feat_sizes[-1], 5, 2, 2))

        self.conv_layers.append(ConvBlock(64, 128, 3))
        conv_feat_sizes.append(compute_conv_output_size(conv_feat_sizes[-1], 3, 2, 1))

        self.conv_layers.append(ConvBlock(128, 256, 3))
        conv_feat_sizes.append(compute_conv_output_size(conv_feat_sizes[-1], 3, 2, 1))

        self.conv_layers.append(ConvBlock(256, 512, 3))
        conv_feat_sizes.append(compute_conv_output_size(conv_feat_sizes[-1], 3, 2, 1))

        self.conv_layers.append(ConvBlock(512, 512, 3))
        conv_feat_sizes.append(compute_conv_output_size(conv_feat_sizes[-1], 3, 2, 1))

        self.conv_layers.append(ConvBlock(512, 512, 3))
        conv_feat_sizes.append(compute_conv_output_size(conv_feat_sizes[-1], 3, 2, 1))

        # print(conv_feat_sizes)

        self.upconv_layers = nn.ModuleList([UpConv(512, 512, 3, 2, tuple(np.mod(conv_feat_sizes[-2], 2) == 0))])
        self.iconv_layers = nn.ModuleList([Conv(512*2, 512, 3, 1, 1)])

        self.upconv_layers.append(UpConv(512, 512, 3, 2, tuple(np.mod(conv_feat_sizes[-3], 2) == 0)))
        self.iconv_layers.append(Conv(512*2, 512, 3, 1, 1))
        self.invdepth_layers = nn.ModuleList([Conv(512, 1, 3, 1, 1, nn.Sigmoid())])

        self.upconv_layers.append(UpConv(512, 256, 3, 2, tuple(np.mod(conv_feat_sizes[-4], 2) == 0)))
        self.iconv_layers.append(Conv(256*2, 256, 3, 1, 1))
        self.invdepth_layers.append(Conv(256, 1, 3, 1, 1, nn.Sigmoid()))

        self.upconv_layers.append(UpConv(256, 128, 3, 2, tuple(np.mod(conv_feat_sizes[-5], 2) == 0)))
        self.iconv_layers.append(Conv(128*2, 128, 3, 1, 1))
        self.invdepth_layers.append(Conv(128, 1, 3, 1, 1, nn.Sigmoid()))

        self.upconv_layers.append(UpConv(128, 64, 3, 2, tuple(np.mod(conv_feat_sizes[-6], 2) == 0)))
        self.iconv_layers.append(Conv(64*2+1, 64, 3, 1, 1))
        self.invdepth_layers.append(Conv(64, 1, 3, 1, 1, nn.Sigmoid()))

        self.upconv_layers.append(UpConv(64, 32, 3, 2, tuple(np.mod(conv_feat_sizes[-7], 2) == 0)))
        self.iconv_layers.append(Conv(32*2+1, 32, 3, 1, 1))
        self.invdepth_layers.append(Conv(32, 1, 3, 1, 1, nn.Sigmoid()))

        self.upconv_layers.append(UpConv(32, 16, 3, 2, (1, 1)))
        self.iconv_layers.append(Conv(16+1, 16, 3, 1, 1))
        self.invdepth_layers.append(Conv(16, 1, 3, 1, 1, nn.Sigmoid()))
        # self.invdepth_layers.append(InvDepth(16))

    def init_weights(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                # m.weight.data.normal_(0.0, 0.02)
                m.bias.data = torch.zeros(m.bias.data.size())

        self.apply(weights_init)

    def forward(self, input):
        conv_feat = self.conv_layers[0].forward(input)
        self.conv_feats = [conv_feat]
        for i in range(1, len(self.conv_layers)):
            conv_feat = self.conv_layers[i].forward(self.conv_feats[i-1])
            self.conv_feats.append(conv_feat)

        upconv_feats = []
        invdepth_pyramid = []
        for i in range(0, len(self.upconv_layers)):
            if i==0:
                x = self.upconv_layers[i].forward(self.conv_feats[-1])
            else:
                x = self.upconv_layers[i].forward(upconv_feats[i-1])
            if i<len(self.upconv_layers)-1:
                if x.size(-1) != self.conv_feats[-2-i].size(-1):
                    x = x[:, :, :, :-1]
                if x.size(-2) != self.conv_feats[-2-i].size(-2):
                    x = x[:, :, :-1, :]

            if i==(len(self.upconv_layers)-1):
                x = torch.cat((x, nn.Upsample(scale_factor=2, mode='bilinear')(invdepth_pyramid[-1])), 1)
            elif i > 3:
                x = torch.cat((x, self.conv_feats[-(2+i)], nn.Upsample(scale_factor=2, mode='bilinear')(invdepth_pyramid[-1])), 1)
            else:
                x = torch.cat((x, self.conv_feats[-(2+i)]), 1)
            upconv_feats.append(self.iconv_layers[i].forward(x))
            if i>0:
                # invdepth_pyramid.append(self.invdepth_layers[i-1].forward(upconv_feats[-1])*DISP_SCALING+MIN_DISP)
                invdepth_pyramid.append(self.invdepth_layers[i-1].forward(upconv_feats[-1]))
                # invdepth_pyramid.append(self.invdepth_layers[i-1].forward(upconv_feats[-1]))
        invdepth_pyramid = invdepth_pyramid[-1::-1]
        invdepth_pyramid = invdepth_pyramid[0:5]
        # conv_feats_output = conv_feats_output[0:5]
        for i in range(len(invdepth_pyramid)):
            invdepth_pyramid[i] = invdepth_pyramid[i].squeeze(1)*DISP_SCALING+MIN_DISP
        return invdepth_pyramid
        # return invdepth_pyramid, invdepth0_pyramid, normalize_convfeat_pyramid(conv_feats_output)

class PoseNet(nn.Module):
    def __init__(self, bundle_size):
        super(PoseNet, self).__init__()
        self.bundle_size = bundle_size

        model = [nn.Conv2d(bundle_size*3, 16, kernel_size=7, stride=2, padding=3, bias=True),
                 nn.ReLU(True),
                 nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=True),
                 nn.ReLU(True),
                 nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
                 nn.ReLU(True),
                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
                 nn.ReLU(True),
                 nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
                 nn.ReLU(True),
                 nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                 nn.ReLU(True),
                 nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                 nn.ReLU(True),
                 nn.Conv2d(256, 6*(bundle_size-1), kernel_size=3, stride=2, padding=1, bias=True)
                 ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        assert(self.bundle_size*3 == input.size(1))
        p = self.model.forward(input)
        p = p.view(input.size(0), 6*(self.bundle_size-1), -1).mean(2)
        return p.view(input.size(0), self.bundle_size-1, 6) * 0.01




if __name__ == "__main__":
    net = PoseNet(3)
    p = net.forward(Variable(torch.rand(1, 9, 128, 416)))
    print(p)
