from collections import namedtuple

import torch.nn as nn
from torchvision import models


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
        x = self.sequential_1(x)
        h_relu_1_2 = x
        x = self.sequential_2(x)
        h_relu_2_2 = x
        x = self.sequential_3(x)
        h_relu_3_3 = x
        x = self.sequential_4(x)
        h_relu_4_3 = x
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.instance1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.instance2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.instance3 = nn.InstanceNorm2d(128, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsample Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.instance4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.instance5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

        # Non-linearity
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.instance1(self.conv1(x)))
        x = self.relu(self.instance2(self.conv2(x)))
        x = self.relu(self.instance3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.relu(self.instance4(self.deconv1(x)))
        x = self.relu(self.instance5(self.deconv2(x)))
        x = self.deconv3(x)
        return x


class ResidualBlock(nn.Module):
    """
    ResidualBlock

    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(num_channels, num_channels, kernel_size=3, stride=1)
        self.instance1 = nn.InstanceNorm2d(num_channels, affine=True)
        self.conv2 = ConvLayer(num_channels, num_channels, kernel_size=3, stride=1)
        self.instance2 = nn.InstanceNorm2d(num_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.instance1(self.conv1(x)))
        x = self.instance2(self.conv2(x))
        x += residual
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2                               # \ add reflection padding instead black padding
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)        # /
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    """
    UpsampleConvLayer:

    Upsample the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample

        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
