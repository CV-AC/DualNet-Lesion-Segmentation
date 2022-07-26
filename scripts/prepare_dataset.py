import os
from os.path import join

import cv2
import numpy as np
import pandas as pd
from PIL import Image

"""
original dataset structure (e.g. IDRiD):
IDRID
|-- image
|   |-- test
|   `-- train
`-- label
    |-- test
    |   `-- EX
    `-- train
        `-- EX
"""

configs = {
    'IDRID': {
        'origin_path': './original_data',
        'output_path': './data',
        'splits': ['train', 'test'],
        'region_crop': None,
        'resize': ((960, 1440), None),
    },
    'DDR': {
        'origin_path': './original_data',
        'output_path': './data',
        'splits': ['train', 'valid', 'test'],
        'region_crop': [(20, False), (30, True), (20, False)],
        'resize': ((1024, 1024), 1024),
    },
}


def get_bound(image, threshold, mode_keep=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.dilate(thresh, kernel)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = [0, 0, 0, 0]
    for cnt in contours:
        res = cv2.boundingRect(cnt)
        if res[2] > result[2] or res[3] > result[3]:
            result = res
    x, y, w, h = result

    if mode_keep:
        if np.abs(image.shape[0] / image.shape[1] - 1.0) < 0.1:
            x, y, w, h = 0, 0, image.shape[1], image.shape[0]

    return x, y, w, h


def pad(image, label, ratio):
    height, width = image.shape[0], image.shape[1]

    if width / height <= ratio:
        h = height
        w = int(ratio * h)
    else:
        w = width
        h = int(w / ratio)

    top = int((h - height) / 2)
    left = int((w - width) / 2)
    bottom = h - height - top
    right = w - width - left

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image, label


def region_crop(image_path, output_info_file=None, label_path=None, thresh=20, mode_keep=False, mode_pad=False):
    lines = []
    for root, dirs, files in os.walk(image_path):
        for i, image_file in enumerate(files):
            image = cv2.imread(join(image_path, image_file))

            label = None
            label_file = None
            if label_path:
                label_file = image_file.replace('.jpg', '.png')
                label = cv2.imread(join(label_path, label_file), flags=0)

            ori_h, ori_w = image.shape[0], image.shape[1]
            x, y, w, h = get_bound(image, thresh, mode_keep)

            h0 = y
            h1 = y + h
            w0 = x
            w1 = x + w

            line = '{} {} {} {} {} {} {}\n'.format(image_file.replace('.jpg', '.png'),
                                                   h0, ori_h - h1, w0, ori_w - w1, h, w)
            lines.append(line)

            image = image[h0:h1, w0:w1]
            if label is not None:
                label = label[h0:h1, w0:w1]
                if mode_pad:
                    image, label = pad(image, label, ratio=1.0)

            cv2.imwrite(join(image_path, image_file), image)
            if label is not None:
                cv2.imwrite(join(label_path, label_file), label)

    if output_info_file:
        with open(output_info_file, mode='w') as f:
            f.writelines(lines)


def resize(path, size=None, long_size=None):
    for file in os.listdir(path):
        img = Image.open(join(path, file))
        h, w = np.asarray(img).shape[:2]

        if long_size is not None:
            ratio = 1. * w / h
            if ratio <= 1.0:
                height = long_size
                width = int(ratio * height)
            else:
                width = long_size
                height = int(width / ratio)
        else:
            height, width = size

        img = img.resize((width, height), Image.ANTIALIAS)
        img.save(join(path, file))


def augment(path):
    for file in os.listdir(path):
        img = Image.open(join(path, file))
        name, suffix = file.split('.')

        img.save(join(path, name + '.' + suffix))
        img.transpose(Image.ROTATE_90).save(join(path, name + '_90.' + suffix))
        img.transpose(Image.ROTATE_180).save(join(path, name + '_180.' + suffix))
        img.transpose(Image.ROTATE_270).save(join(path, name + '_270.' + suffix))
        img.transpose(Image.FLIP_LEFT_RIGHT).save(join(path, name + '_horizontal.' + suffix))
        img.transpose(Image.FLIP_TOP_BOTTOM).save(join(path, name + '_vertical.' + suffix))


def generate_lst(path, split, output_path):
    df = pd.DataFrame(np.arange(0).reshape(0, 1), columns=['addr'])

    files = os.listdir(path)
    if len(files) > 0:
        df1 = pd.DataFrame(np.arange(len(files)).reshape((len(files), 1)), columns=['addr'])
        df1.addr = files

        if split == 'train':
            df2 = pd.DataFrame(np.arange(len(files)).reshape((len(files), 1)), columns=['addr'])
            df2.addr = files
            for i in range(len(df2)):
                df2.addr[i] = df2.addr[i].replace('.jpg', '.png')
            df1.addr = 'image/train/' + df1.addr + ' ' + 'label/train/' + df2.addr
        else:
            df1.addr = 'image/{}/'.format(split) + df1.addr

        frames = [df, df1]
        df = pd.concat(frames)
        df.to_csv(join(output_path, split + '.lst'), columns=['addr'], index=False, header=False)


if __name__ == '__main__':
    for dataset, config in configs.items():
        print(dataset)
        print(config['origin_path'], ' -> ', config['output_path'])
        for idx, split in enumerate(config['splits']):
            image_path = join(config['origin_path'], dataset, 'image', split)
            label_path = join(config['origin_path'], dataset, 'label', split, 'EX')
            image_output_path = join(config['output_path'], dataset, 'image', split)
            label_output_path = join(config['output_path'], dataset, 'label', split)

            os.makedirs(image_path, exist_ok=True)
            os.makedirs(label_path, exist_ok=True)
            os.makedirs(image_output_path, exist_ok=True)
            os.makedirs(label_output_path, exist_ok=True)

            print('{} {}: copy files'.format(dataset, split))
            for file in os.listdir(label_path):
                name = file.replace('_EX', '').split('.')[0]
                image = cv2.imread(join(image_path, name + '.jpg'))
                label = cv2.imread(join(label_path, file), flags=0)
                label[label > 0] = 255
                if split == 'train' and np.sum(label) <= 0:
                    continue
                cv2.imwrite(join(image_output_path, name + '.jpg'), image)
                cv2.imwrite(join(label_output_path, name + '.png'), label)

            crop_info = config['region_crop']
            if crop_info is not None:
                print('{} {}: region crop'.format(dataset, split))
                thresh, mode_keep = crop_info[idx]
                crop_info_file = join(config['output_path'], dataset, 'crop_info_{}.txt'.format(split))
                if split == 'train':
                    region_crop(image_output_path, label_path=label_output_path,
                                thresh=thresh, mode_keep=mode_keep, mode_pad=True)
                else:
                    region_crop(image_output_path, crop_info_file, thresh=thresh, mode_keep=mode_keep)

            print('{} {}: resize'.format(dataset, split))
            size, long_size = config['resize']
            if split == 'train':
                resize(image_output_path, size=size)
                resize(label_output_path, size=size)
            else:
                resize(image_output_path, size=size, long_size=long_size)

            if split == 'train':
                print('{} {}: augmentation'.format(dataset, split))
                augment(image_output_path)
                augment(label_output_path)

            print('{} {}: generate lst file'.format(dataset, split))
            generate_lst(image_output_path, split, join(config['output_path'], dataset))
        print()
