from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable


def var(x, dim=0):
    """
    Calculates variance.
    """
    x_zero_mean = x - x.mean(dim).expand_as(x)
    return x_zero_mean.pow(2).mean(dim)


class MultipleConst(nn.Module):
    def forward(self, data):
        return 255 * data


class GramMatrix(nn.Module):
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        conv_block = list()
        conv_block += [norm_layer(inplanes), nn.ReLU(inplace=True), ConvLayer(inplanes, planes, kernel_size=3, stride=stride),
                       norm_layer(planes), nn.ReLU(inplace=True), ConvLayer(planes, planes, kernel_size=3, stride=1),
                       norm_layer(planes)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        residual = (self.residual_layer(x) if self.downsample is not None else x)
        return residual + self.conv_block(x)


class UpBasicBlock(nn.Module):
    """
    Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBasicBlock, self).__init__()
        self.residual_layer = UpsampleConvLayer(inplanes, planes,
                                                kernel_size=1, stride=1, upsample=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes), nn.ReLU(inplace=True), UpsampleConvLayer(inplanes, planes, kernel_size=3, stride=1, upsample=stride),
                       norm_layer(planes), nn.ReLU(inplace=True), ConvLayer(planes, planes, kernel_size=3, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.residual_layer(x) + self.conv_block(x)


class Bottleneck(nn.Module):
    """
    Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion,
                                            kernel_size=1, stride=stride)
        conv_block = list()
        conv_block += [norm_layer(inplanes), nn.ReLU(inplace=True), nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes), nn.ReLU(inplace=True), ConvLayer(planes, planes, kernel_size=3, stride=stride)]
        conv_block += [norm_layer(planes), nn.ReLU(inplace=True), nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        residual = (self.residual_layer(x) if self.downsample is not None else x)
        return residual + self.conv_block(x)


class UpBottleneck(nn.Module):
    """
    Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBottleneck, self).__init__()
        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
                                                kernel_size=1, stride=1, upsample=stride)
        conv_block = list()
        conv_block += [norm_layer(inplanes), nn.ReLU(inplace=True), nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes), nn.ReLU(inplace=True), UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride)]
        conv_block += [norm_layer(planes), nn.ReLU(inplace=True), nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.residual_layer(x) + self.conv_block(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    """
    UpsampleConvLayer
    Upsample the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class Inspiration(nn.Module):
    """
    Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(self, c, b=1):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.weight = nn.Parameter(torch.FloatTensor(1, c, c), requires_grad=True)
        # non-parameter buffer
        self.G = Variable(torch.FloatTensor(b, c, c), requires_grad=True)
        self.C = c
        self.P = None
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def set_target(self, target):
        self.G = target

    def forward(self, x):
        # input X is a 3D feature map
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
        return torch.bmm(self.P.transpose(1, 2).expand(x.size(0), self.C, self.C), x.view(x.size(0), x.size(1), -1)).view_as(x)

    def __repr__(self):
        return self.__class__.__name__ + '(N x ' + str(self.C) + ')'


class VGG16(nn.Module):
    """
    directly used the pre-trained vgg16 model
    """

    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.sequential_1 = nn.Sequential()
        self.sequential_2 = nn.Sequential()
        self.sequential_3 = nn.Sequential()
        self.sequential_4 = nn.Sequential()
        for x in range(4):
            self.sequential_1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.sequential_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.sequential_3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.sequential_4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.sequential_1(x)
        h_relu_1_2 = h
        h = self.sequential_2(h)
        h_relu_2_2 = h
        h = self.sequential_3(h)
        h_relu_3_3 = h
        h = self.sequential_4(h)
        h_relu_4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


class TransformerNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=6, gpu_ids=None):
        super(TransformerNet, self).__init__()
        if gpu_ids is None:
            gpu_ids = []
        self.gpu_ids = gpu_ids
        self.gram = GramMatrix()

        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4

        model1 = list()
        model1 += [ConvLayer(input_nc, 64, kernel_size=7, stride=1),
                   norm_layer(64),
                   nn.ReLU(inplace=True),
                   block(64, 32, 2, 1, norm_layer),
                   block(32 * expansion, ngf, 2, 1, norm_layer)]
        self.model1 = nn.Sequential(*model1)

        model = list()
        self.ins = Inspiration(ngf * expansion)
        model += [self.model1]
        model += [self.ins]

        for i in range(n_blocks):
            model += [block(ngf * expansion, ngf, 1, None, norm_layer)]

        model += [upblock(ngf * expansion, 32, 2, norm_layer),
                  upblock(32 * expansion, 16, 2, norm_layer),
                  norm_layer(16 * expansion),
                  nn.ReLU(inplace=True),
                  ConvLayer(16 * expansion, output_nc, kernel_size=7, stride=1)]

        self.model = nn.Sequential(*model)

    def set_target(self, x):
        final = self.model1(x)
        generated = self.gram(final)
        self.ins.set_target(generated)

    def forward(self, x):
        return self.model(x)
