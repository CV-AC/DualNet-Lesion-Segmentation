import argparse
import os

import cv2
import numpy as np
from sklearn import metrics

pad_info = None
dataset_config = {
    'IDRID': {
        'test': {
            'path': 'data/IDRID/label/test',
            'pad_info_file': None
        }
    },
    'DDR': {
        'test': {
            'path': 'data/DDR/label/test',
            'pad_info_file': 'data/DDR/crop_info_test.txt'
        },
        'valid': {
            'path': 'data/DDR/label/valid',
            'pad_info_file': 'data/DDR/crop_info_valid.txt'
        }
    }
}


def load_pad_info(pad_info_file):
    if pad_info_file is None:
        return

    global pad_info
    with open(pad_info_file) as f:
        pad_info = dict()
        lines = f.readlines()
        for line in lines:
            item = line.strip().split(' ')
            pad_info[item[0]] = [int(i) for i in item[1:]]


def load_pred(pred_file, label_shape, pred_pad_info=None):
    if pred_pad_info is not None:
        key = os.path.basename(pred_file)
        top, bottom, left, right, h, w = pred_pad_info[key]

        pred = cv2.imread(pred_file, flags=0).astype('float')
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        pred = np.pad(pred, ((top, bottom), (left, right)), mode='constant')

        assert pred.shape == label_shape, f'Mismatch shape {pred.shape} {label_shape}'
    else:
        w, h = label_shape
        pred = cv2.imread(pred_file, flags=0).astype('float')
        pred = cv2.resize(pred, (h, w), interpolation=cv2.INTER_LINEAR)

    return pred


def predict(pred_file, label_file, thresh):
    label = cv2.imread(label_file, flags=0).astype('bool')
    pred = load_pred(pred_file, label.shape, pred_pad_info=pad_info)

    thresh = 256. * thresh
    thresh_pred = (pred > thresh)

    P = np.sum(label).astype('float')
    TP = np.sum(label & thresh_pred).astype('float')
    FP = np.sum(~label & thresh_pred).astype('float')

    return P, TP, FP


def evaluate(pred_path, label_path):
    threshs = np.arange(0, 1.1, 0.1)

    ppv_list = []
    sn_list = []
    for thresh in threshs:
        P = TP = FP = 0
        for root, dirs, files in os.walk(pred_path):
            for i, pred_file in enumerate(files):
                label_file = pred_file
                p, tp, fp = predict(os.path.join(pred_path, pred_file),
                                    os.path.join(label_path, label_file),
                                    thresh)
                P += p
                TP += tp
                FP += fp

        if P > 0 and (TP + FP) > 0:
            ppv = TP * 1. / (TP + FP)
            sn = TP * 1. / P
        else:
            ppv = 1.
            sn = 0.

        ppv_list.append(ppv)
        sn_list.append(sn)

    ppv_list.append(1.)
    sn_list.append(0.)
    aupr = metrics.auc(sn_list, ppv_list)
    return sn_list, ppv_list, aupr


def aupr_evaluate(pred_path, label_path, epoch):
    if epoch is not None:
        print('Evaluate on {} -> {}, epoch {}'.format(pred_path, label_path, epoch))
    else:
        print('Evaluate on {} -> {}'.format(pred_path, label_path))

    SN, PPV, AUPR = evaluate(pred_path, label_path)
    print('AUPR:{:6f}'.format(AUPR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, help='Prediction path')
    parser.add_argument('--dataset', type=str, help='Dataset name', choices=['IDRID', 'DDR'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--epoch', type=int, default=None, help='Evaluate on specific epoch result')
    args = parser.parse_args()

    label_path = dataset_config[args.dataset][args.split]['path']
    load_pad_info(dataset_config[args.dataset][args.split]['pad_info_file'])

    split = None
    if args.epoch is not None:
        split = args.split
        pred_path = os.path.join(args.pred, 'epoch-{}-{}'.format(args.epoch, args.split))
    else:
        pred_path = args.pred

    aupr_evaluate(pred_path, label_path, args.epoch)
