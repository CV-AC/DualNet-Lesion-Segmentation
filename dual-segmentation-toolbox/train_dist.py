import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import networks

from dataset.datasets import LesionDataset
from dual.toolbox_combiner import ToolboxCombiner
from engine import Engine
from loss.criterion import CriterionScaledDice, CriterionScaledCBCE, CriterionScaledFocal
from utils.pyt_utils import extant_file
from utils.utils import load_checkpoint, save_checkpoint

global_iteration = 0


def get_parser():
    parser = argparse.ArgumentParser(description='DualNet with PSPNet and DeepLabV3')

    parser.add_argument('--model', type=str, choices=['pspnet', 'deeplabv3', 'dual_pspnet', 'dual_deeplabv3'])
    parser.add_argument('--dual-p', type=float, default=None, help='Re-balanced positive sample rate in DualNet')
    parser.add_argument('--dual-stages', type=int, default=2, help='Used stages in DualNet')
    parser.add_argument('--loss', type=str, default='dice', choices=['cbce', 'dice', 'focal'], help='Loss name')
    parser.add_argument('--focal-alpha', type=float, default=0.75)
    parser.add_argument('--focal-gamma', type=float, default=2.0)

    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--dataset-test-split', type=str, default='test')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--pretrained', type=str, default='./dual-segmentation-toolbox/networks/resnet50-imagenet.pth',
                        help='Pretrained model path')
    parser.add_argument('--checkpoint', default='', help='Resume from the checkpoint')
    parser.add_argument('--start-iteration', type=int, default=0)
    parser.add_argument('--output', default='../output', help='Output path')
    parser.add_argument('--gpus', type=str, default='0,1', help='CUDA visible devices')
    parser.add_argument('--port', type=int, default=1, help='Port for distribute training')

    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-2, help='Initial learning rate')
    parser.add_argument('--power', type=float, default=0.9, help='Decay parameter to compute the learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Regularisation parameter for L2-loss.')
    parser.add_argument('--num-workers', type=int, default=0, help='Num workers')
    parser.add_argument('--print-frequency', type=int, default=100, help='Number of training steps.')
    parser.add_argument('--save-interval', type=int, default=1, help='save intervals')
    return parser


def inject_dist_default_parser(parser):
    p = parser
    p.add_argument('--devices', default='', help='set data parallel training')
    p.add_argument('--continue', type=extant_file, metavar='FILE', dest='continue_fpath',
                   help='continue from one certain checkpoint')
    p.add_argument('--local_rank', default=0, type=int, help='process rank on node')


def adjust_learning_rate(optimizer, lr, i_iter, max_iter, power):
    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    lr = lr_poly(lr, i_iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def train(model, train_dataset, test_dataset, optimizer, args, engine):
    global global_iteration

    train_loader, train_sampler = engine.get_train_loader(train_dataset)

    test_loader = None
    if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, drop_last=False,
                                 shuffle=False, pin_memory=False)

    num_steps = int(args.epochs * len(train_loader))
    run = True

    criterion_dice = CriterionScaledDice()

    while run:
        epoch = global_iteration // len(train_loader)
        if engine.distributed:
            train_sampler.set_epoch(epoch)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(train_loader)), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        for idx in pbar:
            global_iteration += 1

            images, labels = dataloader.next()
            images = images.cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)

            optimizer.zero_grad()

            lr = adjust_learning_rate(optimizer, args.lr, global_iteration - 1, num_steps,
                                      args.power)
            if args.model.startswith('dual'):
                outs = model(images)
                loss = criterion_dice(outs, labels)
            else:
                loss = model(images, labels)

            reduce_loss = engine.all_reduce_tensor(loss)
            reduce_loss.backward()

            optimizer.step()

            print_str = 'Epoch{}/Iters{}'.format(epoch, global_iteration) \
                        + ' Iter{}/{}:'.format(idx + 1, len(train_loader)) \
                        + ' lr=%.6e' % lr \
                        + ' loss=%.9f' % reduce_loss.item()

            pbar.set_description(print_str, refresh=False)

            if global_iteration % len(train_loader) == 0 or global_iteration >= num_steps:
                if test_loader is not None:
                    print('\n[epoch {}] {} ...'.format(epoch, args.dataset_test_split))
                    test(test_loader, model,
                         os.path.join(args.output, 'epoch-{}-{}'.format(epoch, args.dataset_test_split)), args)

                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    if epoch % args.save_interval == 0:
                        print('\nsave checkpoint ...')
                        save_checkpoint(model, optimizer, global_iteration,
                                        os.path.join(args.output, 'epoch-{}-checkpoint.pt'.format(epoch)))

            if global_iteration >= num_steps:
                run = False
                break


def dual_train(model, train_dataset, test_dataset, optimizer, args, engine, device):
    global global_iteration

    batch_size = args.batch_size // engine.world_size
    loss_num = engine.world_size // 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=args.num_workers, drop_last=True, shuffle=True, pin_memory=False)
    sub_train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=args.num_workers, drop_last=True, shuffle=True, pin_memory=False)

    test_loader = None
    if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
        test_loader = DataLoader(test_dataset, batch_size=1,
                                 num_workers=args.num_workers, drop_last=False, shuffle=False, pin_memory=False)
        test_split = args.dataset_test_split

    num_steps = int(args.epochs * len(train_loader))
    combiner = ToolboxCombiner(model, device, args.epochs, p_rate=args.dual_p)

    run = True
    while run:
        epoch = global_iteration // len(train_loader)
        combiner.reset_epoch(epoch)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(train_loader)), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        sub_dataloader = iter(sub_train_loader)

        for idx in pbar:
            global_iteration += 1
            combiner.reset_global_iteration(global_iteration)

            if engine.distributed and engine.local_rank % 2 == 0:
                images, labels = dataloader.next()
            else:
                images, labels = sub_dataloader.next()

            images = images.cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, args.lr, global_iteration - 1, num_steps,
                                      args.power)

            if engine.distributed and engine.local_rank % 2 == 0:
                loss = combiner.dual_forward(images, labels, feature_cb=True)
            else:
                loss = combiner.dual_forward(images, labels, feature_rb=True)

            reduce_loss = engine.all_reduce_tensor(loss, norm=False)
            reduce_loss = reduce_loss / loss_num
            reduce_loss.backward()
            optimizer.step()

            print_str = 'Epoch{}/Iters{}'.format(epoch, global_iteration) \
                        + ' Iter{}/{}:'.format(idx + 1, len(train_loader)) \
                        + ' lr=%.6e' % lr \
                        + ' loss=%.9f' % reduce_loss.item()

            pbar.set_description(print_str, refresh=False)

            if global_iteration % len(train_loader) == 0 or global_iteration >= num_steps:
                if test_loader is not None:
                    print('\n[epoch {}] test ...'.format(epoch))
                    test(test_loader, model, os.path.join(args.output, 'epoch-{}-{}'.format(epoch, test_split)), args)

                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    if epoch % args.save_interval == 0:
                        print('\nsave checkpoint ...')
                        save_checkpoint(model, optimizer, global_iteration,
                                        os.path.join(args.output, 'epoch-{}-checkpoint.pt'.format(epoch)))

            if global_iteration >= num_steps:
                run = False
                break


def test(test_loader, net, save_dir, args=None):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

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
    global global_iteration

    parser = get_parser()
    inject_dist_default_parser(parser)

    args = parser.parse_args()
    if args.port is not None:
        engine_id = args.port
    else:
        engine_id = args.output.split('/')[-1]

    with Engine(engine_id, custom_parser=parser) as engine:

        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)

        cudnn.benchmark = False

        train_dataset = LesionDataset(name=args.dataset, split='train')
        test_dataset = LesionDataset(name=args.dataset, split=args.dataset_test_split)

        global_iteration = args.start_iteration

        if args.model.startswith('dual'):
            model = eval('networks.{}.Seg_Model'.format(args.model))(
                pretrained_model=args.pretrained, logging_keys=False)
        else:
            if args.loss == 'dice':
                criterion = CriterionScaledDice(stages=args.dual_stages)
            elif args.loss == 'cbce':
                criterion = CriterionScaledCBCE(stages=args.dual_stages)
            elif args.loss == 'focal':
                criterion = CriterionScaledFocal(stages=args.dual_stages,
                                                 alpha=args.focal_alpha, gamma=args.focal_gamma)

            model = eval('networks.{}.Seg_Model'.format(args.model))(
                num_classes=args.num_classes, criterion=criterion,
                pretrained_model=args.pretrained, logging_keys=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        model = engine.data_parallel(model)
        model.train()

        optimizer = optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}],
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.zero_grad()

        if args.checkpoint:
            global_iteration = load_checkpoint(model, optimizer, args.checkpoint)

        if args.model.startswith('dual'):
            dual_train(model, train_dataset, test_dataset, optimizer, args, engine, device)
        else:
            train(model, train_dataset, test_dataset, optimizer, args, engine)


if __name__ == '__main__':
    main()
