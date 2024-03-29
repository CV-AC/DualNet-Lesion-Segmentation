import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import load_dual_model

try:
    from inplace_abn import InPlaceABNSync

    BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')
except:
    InPlaceABNSync = BatchNorm2d = torch.nn.BatchNorm2d
    print('WARNING: inplace_abn is not found')


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1))

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.GroupNorm(num_channels=out_features, num_groups=1)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class BackboneBlock(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(BackboneBlock, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3] - 1, stride=1, dilation=4, multi_grid=(1, 1, 1))

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, affine_par=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.training:
            x_dsn = self.dsn(x)
            x = self.layer4(x)
            return [x, x_dsn]
        else:
            x = self.layer4(x)
            return [x, None]


class HeadBlock(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(HeadBlock, self).__init__()

        self._simulate_make_layer(block, 64, layers[0])
        self._simulate_make_layer(block, 128, layers[1], stride=2)
        self._simulate_make_layer(block, 256, layers[2], stride=1, dilation=2)
        self._simulate_make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))
        self.layer4 = self._make_layer(block, 512, 1, stride=1, dilation=4, multi_grid=(1, 1, 1))

        self.head = nn.Sequential(PSPModule(2048, 512),
                                  nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, affine_par=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def _simulate_make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        self.inplanes = planes * block.expansion
        return 0

    def forward(self, x):
        x = self.layer4(x)
        x = self.head(x)
        return x


class Dual_PSPNet(nn.Module):
    def __init__(self):
        super(Dual_PSPNet, self).__init__()
        self.backbone = BackboneBlock(Bottleneck, [3, 4, 6, 3], 1)
        self.cb_block = HeadBlock(Bottleneck, [3, 4, 6, 3], 1)
        self.rb_block = HeadBlock(Bottleneck, [3, 4, 6, 3], 1)
        self.alpha = 0.5

    def upsample(self, outs, h, w):
        return [F.interpolate(input=out, size=(h, w), mode='bilinear', align_corners=True)
                if out is not None else out for out in outs]

    def forward(self, x, **kwargs):
        h, w = x.size(2), x.size(3)
        x, x_dsn = self.backbone(x)

        if "feature_cb_rb" in kwargs:
            cb = self.cb_block(x)
            rb = self.rb_block(x)
            outs = [cb, rb, x_dsn]
            outs = self.upsample(outs, h, w)
            outs = [torch.sigmoid(r) for r in outs]
            return outs
        if "feature_cb" in kwargs:
            x = self.cb_block(x)
            outs = [x, x_dsn]
            outs = self.upsample(outs, h, w)
            outs = [torch.sigmoid(r) if r is not None else r for r in outs]
            return outs
        elif "feature_rb" in kwargs:
            x = self.rb_block(x)
            outs = [x, x_dsn]
            outs = self.upsample(outs, h, w)
            outs = [torch.sigmoid(r) if r is not None else r for r in outs]
            return outs

        cb_outputs = self.cb_block(x)
        rb_outputs = self.rb_block(x)

        x = torch.add(self.alpha * cb_outputs, (1 - self.alpha) * rb_outputs)
        return [x, x_dsn]


def get_mapping_dict():
    keys = ['layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean',
            'layer4.2.bn1.running_var', 'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias',
            'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight',
            'layer4.2.bn3.bias', 'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var']
    d = dict()
    prefixs = ['cb_block.', 'rb_block.']
    for key in keys:
        d[key] = [key.replace('layer4.2', '{}layer4.0'.format(prefix)) for prefix in prefixs]
    return d


def Seg_Model(pretrained_model=None, logging_keys=True):
    model = Dual_PSPNet()

    if pretrained_model is not None:
        model = load_dual_model(model, pretrained_model, mapping_dict=get_mapping_dict(), logging_keys=logging_keys)

    return model
