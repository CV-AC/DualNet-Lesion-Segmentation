import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import networks

from dataset.datasets import LesionDataset
from engine import Engine
from utils.pyt_utils import extant_file
from utils.utils import load_test_checkpoint


def get_parser():
    parser = argparse.ArgumentParser(description='DualNet with PSPNet and DeepLabV3')

    parser.add_argument('--model', type=str, choices=['pspnet', 'deeplabv3', 'dual_pspnet', 'dual_deeplabv3'])
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--dataset-test-split', type=str, default='test')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--output', default='../output', help='Output path')
    parser.add_argument('--gpus', type=str, default='0', help='CUDA visible devices')
    parser.add_argument('--port', type=int, default=None, help='Port for distribute training')
    parser.add_argument('--num-workers', type=int, default=1, help='Num workers')
    parser.add_argument('--pretrained', type=str, default='./dual-segmentation-toolbox/networks/resnet50-imagenet.pth',
                        help='Pretrained model path')
    return parser


def inject_dist_default_parser(parser):
    p = parser
    p.add_argument('--devices', default='', help='set data parallel training')
    p.add_argument('--continue', type=extant_file, metavar='FILE', dest='continue_fpath',
                   help='continue from one certain checkpoint')
    p.add_argument('--local_rank', default=0, type=int, help='process rank on node')


def test(test_loader, net, save_dir, args=None):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    net.eval()

    for batch_index, images in enumerate(tqdm(test_loader)):
        images = images.cuda()
        _, _, h, w = images.shape

        with torch.no_grad():
            preds_list = net(images)
            scale_pred = torch.nn.functional.interpolate(
                input=preds_list[0], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred = torch.sigmoid(scale_pred)
        fuse = scale_pred.detach().cpu().numpy()[0, 0]
        name = test_loader.dataset.images_name[batch_index]
        Image.fromarray((fuse * 255).astype(np.uint8)).save(os.path.join(save_dir, '{}.png'.format(name)))


def main():
    parser = get_parser()
    inject_dist_default_parser(parser)

    args = parser.parse_args()
    with Engine(custom_parser=parser) as engine:

        if not os.path.exists(args.output):
            os.makedirs(args.output)

        cudnn.benchmark = False

        test_dataset = LesionDataset(name=args.dataset, split=args.dataset_test_split)
        test_loader = DataLoader(test_dataset, batch_size=1,
                                 num_workers=args.num_workers, drop_last=False, shuffle=False, pin_memory=True)

        if args.model.startswith('dual'):
            model = eval('networks.{}.Seg_Model'.format(args.model))(
                pretrained_model=args.pretrained, logging_keys=False)
        else:
            model = eval('networks.{}.Seg_Model'.format(args.model))(
                num_classes=args.num_classes, criterion=None,
                pretrained_model=args.pretrained, logging_keys=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        model = engine.data_parallel(model)
        model.eval()

        if args.checkpoint:
            load_test_checkpoint(model, args.checkpoint)

        test(test_loader, model, args.output, args)


if __name__ == '__main__':
    main()
