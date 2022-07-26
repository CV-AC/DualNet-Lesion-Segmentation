import torch
import torch.nn as nn
from collections import defaultdict
from models.utils import make_bilinear_weights, weights_init
import pickle


class BackboneBlock(nn.Module):
    def __init__(self, device):
        super(BackboneBlock, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=35)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.score_dsn1 = nn.Conv2d(64, 1, 1)
        self.score_dsn2 = nn.Conv2d(128, 1, 1)
        self.score_dsn3 = nn.Conv2d(256, 1, 1)

        self.weight_deconv2 = make_bilinear_weights(4, 1).to(device)
        self.weight_deconv3 = make_bilinear_weights(8, 1).to(device)

        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = \
            prepare_aligned_crop()

    def forward(self, x):
        image_h, image_w = x.shape[2], x.shape[3]

        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.maxpool(conv3_3)

        score_dsn1 = self.score_dsn1(conv1_2)
        score_dsn2 = self.score_dsn2(conv2_2)
        score_dsn3 = self.score_dsn3(conv3_3)

        upsample2 = torch.nn.functional.conv_transpose2d(score_dsn2, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(score_dsn3, self.weight_deconv3, stride=4)

        crop1 = score_dsn1[:, :, self.crop1_margin:self.crop1_margin + image_h,
                self.crop1_margin:self.crop1_margin + image_w]
        crop2 = upsample2[:, :, self.crop2_margin:self.crop2_margin + image_h,
                self.crop2_margin:self.crop2_margin + image_w]
        crop3 = upsample3[:, :, self.crop3_margin:self.crop3_margin + image_h,
                self.crop3_margin:self.crop3_margin + image_w]

        crops = torch.cat((crop1, crop2, crop3), dim=1)
        return pool3, crops


class HeadBlock(nn.Module):
    def __init__(self, device):
        super(HeadBlock, self).__init__()

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.score_dsn4 = nn.Conv2d(512, 1, 1)
        self.score_dsn5 = nn.Conv2d(512, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)

        self.weight_deconv4 = make_bilinear_weights(16, 1).to(device)
        self.weight_deconv5 = make_bilinear_weights(32, 1).to(device)

        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = \
            prepare_aligned_crop()

    def forward(self, x, crops, image_shape):
        image_h, image_w = image_shape

        conv4_1 = self.relu(self.conv4_1(x))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.maxpool(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        score_dsn4 = self.score_dsn4(conv4_3)
        score_dsn5 = self.score_dsn5(conv5_3)

        upsample4 = torch.nn.functional.conv_transpose2d(score_dsn4, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(score_dsn5, self.weight_deconv5, stride=16)

        crop4 = upsample4[:, :, self.crop4_margin:self.crop4_margin + image_h,
                self.crop4_margin:self.crop4_margin + image_w]
        crop5 = upsample5[:, :, self.crop5_margin:self.crop5_margin + image_h,
                self.crop5_margin:self.crop5_margin + image_w]

        fuse_cat = torch.cat((crops, crop4, crop5), dim=1)
        fuse = self.score_final(fuse_cat)
        results = [crop4, crop5, fuse]

        return results


class DUAL_HED(nn.Module):
    def __init__(self, device, alpha=0.5):
        super(DUAL_HED, self).__init__()
        self.backbone = BackboneBlock(device)
        self.cb_block = HeadBlock(device)
        self.rb_block = HeadBlock(device)
        self.alpha = alpha

    def forward(self, x, **kwargs):
        image_h, image_w = x.shape[2], x.shape[3]

        stage3, crops = self.backbone(x)

        if "feature_cb" in kwargs:
            outputs = self.cb_block(stage3, crops, (image_h, image_w))
            outputs = [torch.sigmoid(r) for r in outputs]
            return outputs
        elif "feature_rb" in kwargs:
            outputs = self.rb_block(stage3, crops, (image_h, image_w))
            outputs = [torch.sigmoid(r) for r in outputs]
            return outputs

        cb_outputs = self.cb_block(stage3, crops, (image_h, image_w))
        rb_outputs = self.rb_block(stage3, crops, (image_h, image_w))

        output = torch.add(self.alpha * cb_outputs[-1], (1 - self.alpha) * rb_outputs[-1])
        output = torch.sigmoid(output)
        return [output]


def DUAL_HED_VGG16(device, test_alpha=None):
    if test_alpha:
        net = DUAL_HED(device, test_alpha)
    else:
        net = DUAL_HED(device)
    net.to(device)
    net.apply(weights_init)
    return net


def get_model(device, pretrained=None, test_alpha=None):
    model = DUAL_HED_VGG16(device, test_alpha)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    if pretrained is not None:
        load_vgg16_caffe(model, pretrained)

    return model


def prepare_aligned_crop():
    def map_inv(m):
        """ Mapping inverse. """
        a, b = m
        return 1 / a, -b / a

    def map_compose(m1, m2):
        """ Mapping compose. """
        a1, b1 = m1
        a2, b2 = m2
        return a1 * a2, a1 * b2 + b1

    def deconv_map(kernel_h, stride_h, pad_h):
        """ Deconvolution coordinates mapping. """
        return stride_h, (kernel_h - 1) / 2 - pad_h

    def conv_map(kernel_h, stride_h, pad_h):
        """ Convolution coordinates mapping. """
        return map_inv(deconv_map(kernel_h, stride_h, pad_h))

    def pool_map(kernel_h, stride_h, pad_h):
        """ Pooling coordinates mapping. """
        return conv_map(kernel_h, stride_h, pad_h)

    x_map = (1, 0)
    conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
    conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
    pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

    conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
    conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
    pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

    conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
    conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
    conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
    pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

    conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
    conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
    conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
    pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

    conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
    conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
    conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

    score_dsn1_map = conv1_2_map
    score_dsn2_map = conv2_2_map
    score_dsn3_map = conv3_3_map
    score_dsn4_map = conv4_3_map
    score_dsn5_map = conv5_3_map

    upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
    upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
    upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
    upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

    crop1_margin = int(score_dsn1_map[1])
    crop2_margin = int(upsample2_map[1])
    crop3_margin = int(upsample3_map[1])
    crop4_margin = int(upsample4_map[1])
    crop5_margin = int(upsample5_map[1])

    return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin


def load_vgg16_caffe(net, path='./5stage-vgg.py36pickle'):
    load_pretrained_caffe(net, path, only_vgg=True)


def load_pretrained_caffe(net, path='./hed_pretrained_bsds.py36pickle', only_vgg=False):
    with open(path, 'rb') as f:
        pretrained_params = pickle.load(f)

    print('=> Start loading parameters...')
    vgg_layers_name = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
    for name, param in net.named_parameters():
        _, layer_name, var_name = name.split('.')[-3:]
        if (only_vgg is False) or ((only_vgg is True) and (layer_name in vgg_layers_name)):
            param.data.copy_(torch.from_numpy(pretrained_params[layer_name][var_name]))
            print('=> Loaded {}.'.format(name))
    print('=> Finish loading parameters.')


def get_optimizer(net, lr, momentum, weight_decay):
    net_parameters_id = defaultdict(list)
    for name, param in net.named_parameters():
        _, layer_name, var_name = name.split('.')[-3:]
        if layer_name in ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                          'conv4_1', 'conv4_2', 'conv4_3']:
            if var_name == 'weight':
                print('{:26} lr:    1 decay:1'.format(name))
                net_parameters_id['conv1-4.weight'].append(param)
            elif var_name == 'bias':
                print('{:26} lr:    2 decay:0'.format(name))
                net_parameters_id['conv1-4.bias'].append(param)

        elif layer_name in ['conv5_1', 'conv5_2', 'conv5_3']:
            if var_name == 'weight':
                print('{:26} lr:  100 decay:1'.format(name))
                net_parameters_id['conv5.weight'].append(param)
            elif var_name == 'bias':
                print('{:26} lr:  200 decay:0'.format(name))
                net_parameters_id['conv5.bias'].append(param)

        elif layer_name in ['score_dsn1', 'score_dsn2',
                            'score_dsn3', 'score_dsn4', 'score_dsn5']:
            if var_name == 'weight':
                print('{:26} lr: 0.01 decay:1'.format(name))
                net_parameters_id['score_dsn_1-5.weight'].append(param)
            elif var_name == 'bias':
                print('{:26} lr: 0.02 decay:0'.format(name))
                net_parameters_id['score_dsn_1-5.bias'].append(param)

        elif layer_name in ['score_final']:
            if var_name == 'weight':
                print('{:26} lr:0.001 decay:1'.format(name))
                net_parameters_id['score_final.weight'].append(param)
            elif var_name == 'bias':
                print('{:26} lr:0.002 decay:0'.format(name))
                net_parameters_id['score_final.bias'].append(param)

    opt = torch.optim.SGD([
        {'params': net_parameters_id['conv1-4.weight'], 'lr': lr * 1, 'weight_decay': weight_decay},
        {'params': net_parameters_id['conv1-4.bias'], 'lr': lr * 2, 'weight_decay': 0.},
        {'params': net_parameters_id['conv5.weight'], 'lr': lr * 100, 'weight_decay': weight_decay},
        {'params': net_parameters_id['conv5.bias'], 'lr': lr * 200, 'weight_decay': 0.},
        {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': lr * 0.01, 'weight_decay': weight_decay},
        {'params': net_parameters_id['score_dsn_1-5.bias'], 'lr': lr * 0.02, 'weight_decay': 0.},
        {'params': net_parameters_id['score_final.weight'], 'lr': lr * 0.001, 'weight_decay': weight_decay},
        {'params': net_parameters_id['score_final.bias'], 'lr': lr * 0.002, 'weight_decay': 0.},
    ], lr=lr, momentum=momentum, weight_decay=weight_decay)

    return opt
