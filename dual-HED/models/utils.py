import torch
import torch.nn as nn
import numpy as np


def make_bilinear_weights(size, num_channels):
    """ Generate bi-linear interpolation weights as up-sampling filters (following FCN paper). """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False  # Set not trainable.
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def weights_init(m, channels=5):
    """ Weight initialization function. """
    if isinstance(m, nn.Conv2d):
        # Initialize: m.weight.
        if m.weight.data.shape == torch.Size([1, channels, 1, 1]):
            # Constant initialization for fusion layer in HED models.
            torch.nn.init.constant_(m.weight, 0.2)
        else:
            # Zero initialization following official repository.
            # Reference: hed/docs/tutorial/layers.md
            m.weight.data.zero_()
        # Initialize: m.bias.
        if m.bias is not None:
            # Zero initialization.
            m.bias.data.zero_()


def tensor_center_crop(x, output_shape):
    image_height, image_width = x.shape[2], x.shape[3]
    crop_height, crop_width = output_shape[2], output_shape[3]

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))

    return x[:, :, crop_top:crop_top + crop_height, crop_left:crop_left + crop_width]
