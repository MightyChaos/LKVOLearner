import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np


DISP_SCALING = 10
MIN_DISP = 0.01

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
    def __init__(self, input_nc, output_nc, kernel_size):
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


class VggDepthEstimator(nn.Module):
    def __init__(self, input_size=None):
        super(VggDepthEstimator, self).__init__()
        self.conv_layers = nn.ModuleList([ConvBlock(3, 32, 7)])
        self.conv_layers.append(ConvBlock(32, 64, 5))
        self.conv_layers.append(ConvBlock(64, 128, 3))

        self.conv_layers.append(ConvBlock(128, 256, 3))

        self.conv_layers.append(ConvBlock(256, 512, 3))


        self.conv_layers.append(ConvBlock(512, 512, 3))


        self.conv_layers.append(ConvBlock(512, 512, 3))


        # print(conv_feat_sizes)

        self.upconv_layers = nn.ModuleList([UpConv(512, 512, 3)])
        self.iconv_layers = nn.ModuleList([Conv(512*2, 512, 3, 1, 1)])

        self.upconv_layers.append(UpConv(512, 512, 3))
        self.iconv_layers.append(Conv(512*2, 512, 3, 1, 1))
        self.invdepth_layers = nn.ModuleList([Conv(512, 1, 3, 1, 1, nn.Sigmoid())])

        self.upconv_layers.append(UpConv(512, 256, 3))
        self.iconv_layers.append(Conv(256*2, 256, 3, 1, 1))
        self.invdepth_layers.append(Conv(256, 1, 3, 1, 1, nn.Sigmoid()))

        self.upconv_layers.append(UpConv(256, 128, 3))
        self.iconv_layers.append(Conv(128*2, 128, 3, 1, 1))
        self.invdepth_layers.append(Conv(128, 1, 3, 1, 1, nn.Sigmoid()))

        self.upconv_layers.append(UpConv(128, 64, 3))
        self.iconv_layers.append(Conv(64*2+1, 64, 3, 1, 1))
        self.invdepth_layers.append(Conv(64, 1, 3, 1, 1, nn.Sigmoid()))

        self.upconv_layers.append(UpConv(64, 32, 3))
        self.iconv_layers.append(Conv(32*2+1, 32, 3, 1, 1))
        self.invdepth_layers.append(Conv(32, 1, 3, 1, 1, nn.Sigmoid()))

        self.upconv_layers.append(UpConv(32, 16, 3))
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


class PoseExpNet(nn.Module):
    def __init__(self, bundle_size):
        super(PoseExpNet, self).__init__()
        self.bundle_size = bundle_size
        self.convlyr1 = nn.Sequential(*[nn.Conv2d(bundle_size*3, 16, kernel_size=7, stride=2, padding=3, bias=True),
                 nn.ReLU(True)])
        self.convlyr2 = nn.Sequential(*[nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=True),
                 nn.ReLU(True)])
        self.convlyr3 = nn.Sequential(*[nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
                 nn.ReLU(True)])
        self.convlyr4 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
                 nn.ReLU(True)])
        self.convlyr5 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
                 nn.ReLU(True)])

        self.poselyr = nn.Sequential(*[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.ReLU(True),
                        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.ReLU(True),
                        nn.Conv2d(256, 6*(bundle_size-1), kernel_size=3, stride=2, padding=1, bias=True)])

        self.uplyr5 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                          out_channels=256,
                                          kernel_size=2,
                                          bias=True,
                                          stride=2,
                                          padding=0),
                       nn.ReLU(True)])
        self.uplyr4 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,
                                          out_channels=128,
                                          kernel_size=2,
                                          bias=True,
                                          stride=2,
                                          padding=0),
                       nn.ReLU(True)])
        self.uplyr3 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128,
                                          out_channels=64,
                                          kernel_size=2,
                                          bias=True,
                                          stride=2,
                                          padding=0),
                                    nn.ReLU(True)])
        self.uplyr2 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=64,
                                          out_channels=32,
                                          kernel_size=2,
                                          bias=True,
                                          stride=2,
                                          padding=0),
                                          nn.ReLU(True)])
        self.uplyr1 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=32,
                                          out_channels=16,
                                          kernel_size=2,
                                          bias=True,
                                          stride=2,
                                          padding=0),
                                          nn.ReLU(True)])
        self.explyr4 = nn.Sequential(*[nn.Conv2d(128, bundle_size, kernel_size=3,
                                        stride=1, padding=1, bias=True),
                       nn.Sigmoid()])
        self.explyr3 = nn.Sequential(*[nn.Conv2d(64, bundle_size, kernel_size=3,
                                        stride=1, padding=1, bias=True),
                       nn.Sigmoid()])
        self.explyr2 = nn.Sequential(*[nn.Conv2d(32, bundle_size, kernel_size=3,
                                        stride=1, padding=1, bias=True),
                       nn.Sigmoid()])
        self.explyr1 = nn.Sequential(*[nn.Conv2d(16, bundle_size, kernel_size=3,
                                        stride=1, padding=1, bias=True),
                       nn.Sigmoid()])

    def forward(self, input):
        conv1 = self.convlyr1(input)
        conv2 = self.convlyr2(conv1)
        conv3 = self.convlyr3(conv2)
        conv4 = self.convlyr4(conv3)
        conv5 = self.convlyr5(conv4)

        # output pose
        p = self.poselyr.forward(conv5)
        p = p.view(input.size(0), 6*(self.bundle_size-1), -1).mean(2)
        # multiply predicted pose with a small constant
        p = p.view(input.size(0), self.bundle_size-1, 6) * 0.01
        # predict multi-scale explainable mask
        upcnv5 = self.uplyr5(conv5)
        upcnv4 = self.uplyr4(upcnv5)
        upcnv3 = self.uplyr3(upcnv4)
        upcnv2 = self.uplyr2(upcnv3)
        upcnv1 = self.uplyr1(upcnv2)

        mask4 = self.explyr4(upcnv4)
        mask3 = self.explyr3(upcnv3)
        mask2 = self.explyr2(upcnv2)
        mask1 = self.explyr1(upcnv1)

        return p, [mask1, mask2, mask3, mask4]



if __name__ == "__main__":
    model = PoseExpNet(3).cuda()
    x = Variable(torch.randn(1,9,128,416).cuda())
    p, masks = model.forward(x)
    for i in range(4):
        print(masks[i].size())


    dnet = VggDepthEstimator([128,416]).cuda()
    I = Variable(torch.randn(1,3,128,416).cuda())
    invdepth_pyramid = dnet.forward(I)
    for i in range(len(invdepth_pyramid)):
        print(invdepth_pyramid[i].size())
