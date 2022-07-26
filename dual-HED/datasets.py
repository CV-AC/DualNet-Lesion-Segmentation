import os

import cv2
import numpy as np
from torch.utils import data


class LesionDataset(data.Dataset):
    configs = {
        'DDR': {
            'path': './data/DDR',
            'means': (21.216, 50.636, 81.205),
        },
        'IDRID': {
            'path': './data/IDRID',
            'means': (16.423, 56.430, 116.540),
        }
    }

    def __init__(self, name='IDRID', split='train'):
        self.dataset_path = self.configs[name]['path']
        self.means = self.configs[name]['means']
        self.split = split

        self.list_path = os.path.join(self.dataset_path, '{}.lst'.format(split))

        with open(self.list_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        if self.split == 'train':
            pairs = [line.split() for line in lines]
            self.images_path = [pair[0] for pair in pairs]
            self.labels_path = [pair[1] for pair in pairs]
        else:
            self.images_path = lines
            self.images_name = []
            for path in self.images_path:
                folder, filename = os.path.split(path)
                name, ext = os.path.splitext(filename)
                self.images_name.append(name)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        label = None
        if self.split == "train":
            label_path = os.path.join(self.dataset_path, self.labels_path[index])
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = label[np.newaxis, :, :]
            label[label < 127.5] = 0.0
            label[label >= 127.5] = 1.0
            label = label.astype(np.float32)

        image_path = os.path.join(self.dataset_path, self.images_path[index])
        image = cv2.imread(image_path).astype(np.float32)
        image = image - np.array(self.means)
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)

        if self.split == 'train':
            return image, label
        else:
            return image
