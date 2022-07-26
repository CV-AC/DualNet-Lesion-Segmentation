import argparse
import os
import sys

import cupy as cp
import numpy as np
import cv2
from skimage import measure

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


def predict(pred_file, label_file, thresh=0.5, inter_thresh=0.2):
    thresh = 256. * thresh

    label = cv2.imread(label_file, flags=0).astype('bool')
    pred = load_pred(pred_file, label.shape, pred_pad_info=pad_info)

    w, h = label.shape
    pred = (pred > thresh)

    pred_bw = cp.array(measure.label(pred, connectivity=2))
    label_bw = cp.array(measure.label(label, connectivity=2))

    label_mask = cp.array(label)
    pred_mask = cp.array(pred)

    tp = cp.zeros(label_bw.shape).astype('bool')
    tp = tp | pred_mask & label_mask

    for i in range(1, cp.asnumpy(cp.max(pred_bw)) + 1):
        D = pred_bw == i
        I = D & label_mask

        n_i = cp.sum(I)
        n_d = cp.sum(D)

        if n_i / n_d > inter_thresh:
            tp = tp | D

    for i in range(1, cp.asnumpy(cp.max(label_bw)) + 1):
        G = label_bw == i
        I = G & pred_mask

        n_i = cp.sum(I)
        n_g = cp.sum(G)

        if n_i / n_g > inter_thresh:
            tp = tp | G

    fp = pred_mask & ~tp
    fn = label_mask & ~tp

    tp = cp.asnumpy(cp.sum(tp))
    fn = cp.asnumpy(cp.sum(fn))
    fp = cp.asnumpy(cp.sum(fp))
    tn = h * w - tp - fp - fn

    return tp, fn, fp, tn


def evaluate(pred_path, label_path, thresh, inter_thresh):
    TP = FN = FP = TN = 0
    for root, dirs, files in os.walk(pred_path):
        for i, pred_file in enumerate(files):
            label_file = pred_file
            tp, fn, fp, tn = predict(os.path.join(pred_path, pred_file),
                                     os.path.join(label_path, label_file),
                                     thresh=thresh,
                                     inter_thresh=inter_thresh)

            TP += tp
            FN += fn
            FP += fp

    SN = TP / (TP + FN)
    PPV = TP / (TP + FP)
    FSCORE = (SN * PPV * 2) / (SN + PPV)

    return PPV, SN, FSCORE


def region_evaluate(pred_path, label_path, threshs):
    print('Evaluate on {} -> {} '.format(args.pred, label_path))

    summary_txt = []
    for thresh in threshs:
        try:
            results = evaluate(pred_path, label_path, 0.5, inter_thresh=thresh)
            PPV, SN, FSCORE = results
            print('[thresh {:2f}]  PPV:{:6f}, SN:{:6f}, F:{:6f}'.format(
                thresh, PPV, SN, FSCORE))
            summary_txt.append(f'{FSCORE:6f}')
        except KeyboardInterrupt:
            sys.exit(0)
    print('Results:' + ' '.join(summary_txt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, help='Prediction path')
    parser.add_argument('--dataset', type=str, help='Dataset name', choices=['IDRID', 'DDR'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--epoch', type=int, default=None, help='Evaluate on specific epoch result')
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--threshs', type=str, default='0.2,0.35,0.5,0.65,0.8')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    label_path = dataset_config[args.dataset][args.split]['path']
    load_pad_info(dataset_config[args.dataset][args.split]['pad_info_file'])

    epoch = split = None
    if args.epoch is not None:
        epoch = args.epoch
        split = args.split
        pred_path = os.path.join(args.pred, 'epoch-{}-{}'.format(epoch, split))
    else:
        pred_path = args.pred

    threshs = [float(t) for t in args.threshs.split(',')]
    region_evaluate(pred_path, label_path, threshs)
