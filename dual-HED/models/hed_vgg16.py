import torch
import torch.nn as nn
from collections import defaultdict
from models.utils import make_bilinear_weights, weights_init
import pickle


class HED(nn.Module):
    """ HED models. """

    def __init__(self, device):
        super(HED, self).__init__()
        # Layers.
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=35)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.relu = nn.ReLU()
        # Note: ceil_mode – when True, will use ceil instead of floor to compute the output shape.
        #       The reason to use ceil mode here is that later we need to upsample the feature maps and crop the results
        #       in order to have the same shape as the original image. If ceil mode is not used, the up-sampled feature
        #       maps will possibly be smaller than the original images.
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.score_dsn1 = nn.Conv2d(64, 1, 1)  # Out channels: 1.
        self.score_dsn2 = nn.Conv2d(128, 1, 1)
        self.score_dsn3 = nn.Conv2d(256, 1, 1)
        self.score_dsn4 = nn.Conv2d(512, 1, 1)
        self.score_dsn5 = nn.Conv2d(512, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)

        # Fixed bilinear weights.
        self.weight_deconv2 = make_bilinear_weights(4, 1).to(device)
        self.weight_deconv3 = make_bilinear_weights(8, 1).to(device)
        self.weight_deconv4 = make_bilinear_weights(16, 1).to(device)
        self.weight_deconv5 = make_bilinear_weights(32, 1).to(device)

        # Prepare for aligned crop.
        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = \
            self.prepare_aligned_crop()
        # print(self.prepare_aligned_crop())

    def prepare_aligned_crop(self):
        """ Prepare for aligned crop. """

        # Re-implement the logic in deploy.prototxt and
        #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
        # Other reference materials:
        #   hed/include/caffe/layer.hpp
        #   hed/include/caffe/vision_layers.hpp
        #   hed/include/caffe/util/coords.hpp
        #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

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

    def forward(self, x):
        # VGG-16 models.
        image_h, image_w = x.shape[2], x.shape[3]
        # print(image_h, image_w)

        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))  # Side output 1.
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))  # Side output 2.
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))  # Side output 3.
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))  # Side output 4.
        pool4 = self.maxpool(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))  # Side output 5.

        score_dsn1 = self.score_dsn1(conv1_2)
        score_dsn2 = self.score_dsn2(conv2_2)
        score_dsn3 = self.score_dsn3(conv3_3)
        score_dsn4 = self.score_dsn4(conv4_3)
        score_dsn5 = self.score_dsn5(conv5_3)

        upsample2 = torch.nn.functional.conv_transpose2d(score_dsn2, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(score_dsn3, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(score_dsn4, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(score_dsn5, self.weight_deconv5, stride=16)

        # Aligned cropping.
        crop1 = score_dsn1[:, :, self.crop1_margin:self.crop1_margin + image_h,
                self.crop1_margin:self.crop1_margin + image_w]
        crop2 = upsample2[:, :, self.crop2_margin:self.crop2_margin + image_h,
                self.crop2_margin:self.crop2_margin + image_w]
        crop3 = upsample3[:, :, self.crop3_margin:self.crop3_margin + image_h,
                self.crop3_margin:self.crop3_margin + image_w]
        crop4 = upsample4[:, :, self.crop4_margin:self.crop4_margin + image_h,
                self.crop4_margin:self.crop4_margin + image_w]
        crop5 = upsample5[:, :, self.crop5_margin:self.crop5_margin + image_h,
                self.crop5_margin:self.crop5_margin + image_w]

        # Concatenate according to channels.
        fuse_cat = torch.cat((crop1, crop2, crop3, crop4, crop5), dim=1)
        fuse = self.score_final(fuse_cat)  # Shape: [batch_size, 1, image_h, image_w].
        results = [crop1, crop2, crop3, crop4, crop5, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results


def HED_VGG16(device):
    net = HED(device)
    net.to(device)
    net.apply(weights_init)
    return net


def get_model(device, pretrained=None):
    model = HED_VGG16(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    if pretrained is not None:
        load_vgg16_caffe(model, pretrained)

    return model


def load_vgg16_caffe(net, path='./5stage-vgg.py36pickle'):
    """ Load models parameters from VGG-16 Caffe models. """
    load_pretrained_caffe(net, path, only_vgg=True)


def load_pretrained_caffe(net, path='./hed_pretrained_bsds.py36pickle', only_vgg=False):
    """ Load models parameters from pre-trained HED Caffe models. """
    # Read pretrained parameters.
    import os
    print(os.getcwd())
    with open(path, 'rb') as f:
        pretrained_params = pickle.load(f)

    # Load parameters into models.
    print('=> Start loading parameters...')
    vgg_layers_name = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
    for name, param in net.named_parameters():
        print(name)
        _, layer_name, var_name = name.split('.')
        if (only_vgg is False) or ((only_vgg is True) and (layer_name in vgg_layers_name)):
            param.data.copy_(torch.from_numpy(pretrained_params[layer_name][var_name]))
            print('=> Loaded {}.'.format(name))
    print('=> Finish loading parameters.')


def get_optimizer(net, lr, momentum, weight_decay):
    # Optimizer settings.
    net_parameters_id = defaultdict(list)
    for name, param in net.named_parameters():
        if name in ['module.conv1_1.weight', 'module.conv1_2.weight',
                    'module.conv2_1.weight', 'module.conv2_2.weight',
                    'module.conv3_1.weight', 'module.conv3_2.weight', 'module.conv3_3.weight',
                    'module.conv4_1.weight', 'module.conv4_2.weight', 'module.conv4_3.weight']:
            print('{:26} lr:    1 decay:1'.format(name))
            net_parameters_id['conv1-4.weight'].append(param)
        elif name in ['module.conv1_1.bias', 'module.conv1_2.bias',
                      'module.conv2_1.bias', 'module.conv2_2.bias',
                      'module.conv3_1.bias', 'module.conv3_2.bias', 'module.conv3_3.bias',
                      'module.conv4_1.bias', 'module.conv4_2.bias', 'module.conv4_3.bias']:
            print('{:26} lr:    2 decay:0'.format(name))
            net_parameters_id['conv1-4.bias'].append(param)
        elif name in ['module.conv5_1.weight', 'module.conv5_2.weight', 'module.conv5_3.weight']:
            print('{:26} lr:  100 decay:1'.format(name))
            net_parameters_id['conv5.weight'].append(param)
        elif name in ['module.conv5_1.bias', 'module.conv5_2.bias', 'module.conv5_3.bias']:
            print('{:26} lr:  200 decay:0'.format(name))
            net_parameters_id['conv5.bias'].append(param)
        elif name in ['module.score_dsn1.weight', 'module.score_dsn2.weight',
                      'module.score_dsn3.weight', 'module.score_dsn4.weight', 'module.score_dsn5.weight']:
            print('{:26} lr: 0.01 decay:1'.format(name))
            net_parameters_id['score_dsn_1-5.weight'].append(param)
        elif name in ['module.score_dsn1.bias', 'module.score_dsn2.bias',
                      'module.score_dsn3.bias', 'module.score_dsn4.bias', 'module.score_dsn5.bias']:
            print('{:26} lr: 0.02 decay:0'.format(name))
            net_parameters_id['score_dsn_1-5.bias'].append(param)
        elif name in ['module.score_final.weight']:
            print('{:26} lr:0.001 decay:1'.format(name))
            net_parameters_id['score_final.weight'].append(param)
        elif name in ['module.score_final.bias']:
            print('{:26} lr:0.002 decay:0'.format(name))
            net_parameters_id['score_final.bias'].append(param)

    # Create optimizer.
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
