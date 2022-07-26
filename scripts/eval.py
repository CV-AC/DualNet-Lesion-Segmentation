import argparse
import os
import sys

import cv2
import numpy as np

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


def predict(pred_file, label_file, thresh=0.5):
    label = cv2.imread(label_file, flags=0).astype('bool')
    pred = load_pred(pred_file, label.shape, pred_pad_info=pad_info)

    w, h = label.shape
    thresh = 256. * thresh
    pred = (pred > thresh)

    tp = np.sum(label & pred)
    fn = np.sum(label & ~pred)
    fp = np.sum(~label & pred)
    tn = np.sum(w * h - tp - fn - fp)

    return tp, fn, fp, tn


def evaluate(pred_path, label_path, thresh):
    TP = FN = FP = TN = 0
    for root, dirs, files in os.walk(pred_path):
        for i, pred_file in enumerate(files):
            label_file = pred_file
            tp, fn, fp, tn = predict(os.path.join(pred_path, pred_file),
                                     os.path.join(label_path, label_file),
                                     thresh=thresh)
            TP += tp
            FN += fn
            FP += fp
            TN += tn

    PPV = TP / (TP + FP)
    SN = TP / (TP + FN)
    FSCORE = (SN * PPV * 2) / (SN + PPV)
    IOU = TP / (TP + FP + FN)
    return PPV, SN, FSCORE, IOU


def epochs_evaluate(pred_root, label_path, epochs, thresh, split='test'):
    print('Evaluate on {} -> {} '.format(pred_root, label_path))

    best_epoch = 0
    best_metric = 0
    best_results = None
    for epoch in epochs:
        pred_path = os.path.join(pred_root, 'epoch-{}-{}'.format(epoch, split))

        try:
            results = evaluate(pred_path, label_path, thresh)
            PPV, SN, FSCORE, IOU = results
            print('[Epoch{:2d}] PPV:{:6f}, SN:{:6f}, F:{:6f}, IOU:{:6f}'.format(epoch, PPV, SN, FSCORE, IOU))

            if not np.isnan(FSCORE) and FSCORE > best_metric:
                best_epoch = epoch
                best_metric = FSCORE
                best_results = results
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            continue

    if best_results is not None:
        PPV, SN, FSCORE, IOU = best_results
        print('\n[Best {:2d}] PPV:{:6f}, SN:{:6f}, F:{:6f}, IOU:{:6f}'.format(best_epoch, PPV, SN, FSCORE, IOU))


def single_evaluate(pred_path, label_path, thresh):
    print('Evaluate on {} -> {} '.format(pred_path, label_path))
    results = evaluate(pred_path, label_path, thresh)
    PPV, SN, FSCORE, IOU = results
    print('PPV:{:6f}, SN:{:6f}, F:{:6f}, IOU:{:6f}'.format(PPV, SN, FSCORE, IOU))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, help='Prediction path')
    parser.add_argument('--dataset', type=str, help='Dataset name', choices=['IDRID', 'DDR'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=None, help='Evaluate on results of epoch dirs')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=None, help='Evaluate on specific epoch result')
    args = parser.parse_args()

    label_path = dataset_config[args.dataset][args.split]['path']
    load_pad_info(dataset_config[args.dataset][args.split]['pad_info_file'])

    epochs = None
    if args.epoch is not None:
        epochs = [args.epoch]
    elif args.epochs is not None:
        epochs = list(range(0, args.epochs, args.step)) + [args.epochs - 1]

    if epochs is None:
        single_evaluate(args.pred, label_path, args.thresh)
    else:
        epochs_evaluate(args.pred, label_path, epochs, args.thresh, split=args.split)
