# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG

ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg11_bn': ((0, 4), (4, 8), (8, 15), (15, 22), (22, 29)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg13_bn': ((0, 7), (7, 14), (14, 21), (21, 28), (28, 35)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg16_bn': ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37)),
    'vgg19_bn': ((0, 7), (7, 14), (14, 27), (27, 40), (40, 53))
}

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def conv_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
    )


def conv_relu_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False,
                 batch_norm=False):
        super().__init__(make_layers(cfg[model.replace('_bn', '')], batch_norm))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x

        return output




def conv_relu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


def conv_bn_relu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


cfg_eff = {
    'b0': [32, 16, 24, 40, 80, 112, 192, 320, 1280],
    'b1': [32, 16, 24, 40, 80, 112, 192, 320, 1280],
    'b2': [32, 16, 24, 48, 88, 120, 208, 352, 1408],
    'b3': [40, 24, 32, 48, 96, 136, 232, 384, 1536],
    'b4': [48, 24, 32, 56, 112, 160, 272, 448, 1792],
    'b5': [48, 24, 40, 64, 128, 176, 304, 512, 2048],
    'b6': [56, 32, 40, 72, 144, 200, 344, 576, 2304],
    'b7': [64, 32, 48, 80, 160, 224, 384, 640, 2560],
}


class SiameseFCN_damage(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, backbone='vgg16_bn', pretrained=True, requires_grad=True, remove_fc=True,
                 shared=False, fused_method='diff'):
        """
        SiameseFCN for damage assessment
        :param in_ch: input channel number
        :param out_ch: output channel number
        :param backbone: ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
        :param pretrained: default=True
        :param requires_grad: default=True
        :param remove_fc: whether remove the fully connection layer from pretrained model, default=True
        :param shared: whether share the encoder part between two branches
        :param fused_method: 'diff'=pre-post, 'add'=pre+post, 'stack'=concatenate(pre,post)
        """
        super().__init__()

        str_shared = '_shared' if shared else ''
        str_fused_method = '_' + fused_method
        self.name = "Si_FCN_Dam_" + backbone + str_fused_method + str_shared
        self.shared = shared
        self.fused_method = fused_method
        assert fused_method in ['diff', 'add', 'stack']
        assert backbone in ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
        self.pretrained_net1 = VGGNet(pretrained=pretrained, model=backbone, requires_grad=requires_grad,
                                      remove_fc=remove_fc,
                                      batch_norm='bn' in backbone)
        if not shared:
            self.pretrained_net2 = VGGNet(pretrained=pretrained, model=backbone, requires_grad=requires_grad,
                                          remove_fc=remove_fc,
                                          batch_norm='bn' in backbone)
        # input channel!=3
        if in_ch != 3:
            self.pretrained_net1.features[0] = nn.Conv2d(in_ch, 64, 3, 1, 1)
            if not shared:
                self.pretrained_net2.features[0] = nn.Conv2d(in_ch, 64, 3, 1, 1)

        if self.fused_method == 'stack':
            self.conv1 = conv_relu_bn(1024, 512)
            self.conv2 = conv_relu_bn(1024, 512)
            self.conv3 = conv_relu_bn(512, 256)
            self.conv4 = conv_relu_bn(256, 128)
            self.conv5 = conv_relu_bn(128, 64)

        self.relu = nn.ReLU(inplace=True)
        self.deconv1_BD = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv1_D = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1_BD = nn.BatchNorm2d(512)
        self.bn1_D = nn.BatchNorm2d(512)
        self.deconv2_BD = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2_D = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2_BD = nn.BatchNorm2d(256)
        self.bn2_D = nn.BatchNorm2d(256)
        self.deconv3_BD = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3_D = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3_BD = nn.BatchNorm2d(128)
        self.bn3_D = nn.BatchNorm2d(128)
        self.deconv4_BD = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4_D = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4_BD = nn.BatchNorm2d(64)
        self.bn4_D = nn.BatchNorm2d(64)
        self.deconv5_BD = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv5_D = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5_BD = nn.BatchNorm2d(32)
        self.bn5_D = nn.BatchNorm2d(32)
        self.classifier_BD = nn.Conv2d(32, 1, kernel_size=1)
        self.classifier1 = nn.Conv2d(32, out_ch, kernel_size=1)


    def forward_once(self, x):
        output = self.pretrained_net1(x)
        return output

    def forward_seperate(self, x):
        output = self.pretrained_net2(x)
        return output

    def forward(self, data):
        data_pre = data[:, :3, :, :]
        data_post = data[:, 3:, :, :]
        if self.shared:
            output_pre = self.forward_once(data_pre)
            output_post = self.forward_once(data_post)
        else:
            output_pre = self.forward_once(data_pre)
            output_post = self.forward_seperate(data_post)

        x5_pre = output_pre['x5']
        x4_pre = output_pre['x4']
        x3_pre = output_pre['x3']
        x2_pre = output_pre['x2']
        x1_pre = output_pre['x1']

        x5_post = output_post['x5']
        x4_post = output_post['x4']
        x3_post = output_post['x3']
        x2_post = output_post['x2']
        x1_post = output_post['x1']

        if self.fused_method == 'stack':
            x5 = torch.cat((x5_pre, x5_post), 1)
            x4 = torch.cat((x4_pre, x4_post), 1)
            x3 = torch.cat((x3_pre, x3_post), 1)
            x2 = torch.cat((x2_pre, x2_post), 1)
            x1 = torch.cat((x1_pre, x1_post), 1)
            x5 = self.conv1(x5)
            x4 = self.conv2(x4)
            x3 = self.conv3(x3)
            x2 = self.conv4(x2)
            x1 = self.conv5(x1)
        elif self.fused_method == 'diff':
            x5 = x5_pre - x5_post
            x4 = x4_pre - x4_post
            x3 = x3_pre - x3_post
            x2 = x2_pre - x2_post
            x1 = x1_pre - x1_post
        elif self.fused_method == 'add':
            x5 = x5_pre + x5_post
            x4 = x4_pre + x4_post
            x3 = x3_pre + x3_post
            x2 = x2_pre + x2_post
            x1 = x1_pre + x1_post

        # Decoder for building seg
        score_BD = self.bn1_BD(self.relu(self.deconv1_BD(x5_pre)))
        score_BD = score_BD + x4_pre
        score_BD = self.bn2_BD(self.relu(self.deconv2_BD(score_BD)))
        score_BD = score_BD + x3_pre
        score_BD = self.bn3_BD(self.relu(self.deconv3_BD(score_BD)))
        score_BD = score_BD + x2_pre
        score_BD = self.bn4_BD(self.relu(self.deconv4_BD(score_BD)))
        score_BD = score_BD + x1_pre
        score_BD = self.bn5_BD(self.relu(self.deconv5_BD(score_BD)))
        score_BD = self.classifier_BD(score_BD)

        # Decoder for damage seg
        score_D = self.bn1_D(self.relu(self.deconv1_D(x5)))
        score_D = score_D + x4
        score_D = self.bn2_D(self.relu(self.deconv2_D(score_D)))
        score_D = score_D + x3
        score_D = self.bn3_D(self.relu(self.deconv3_D(score_D)))
        score_D = score_D + x2
        score_D = self.bn4_D(self.relu(self.deconv4_D(score_D)))
        score_D = score_D + x1
        score_D = self.bn5_D(self.relu(self.deconv5_D(score_D)))
        score_D = self.classifier1(score_D)
        outputs={}
        outputs["damage"] = score_D
        outputs["building"] = score_BD

        return outputs

if __name__ == "__main__":
    model = SiameseFCN_damage(in_ch=3, out_ch=5, backbone='vgg16_bn', pretrained=True, shared=False,
                              fused_method='diff')
    print(model)

