import os
import sys
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from os.path import join, isdir, abspath, dirname

from datasets import LesionDataset
from utils import Logger, AverageMeter, load_checkpoint, save_checkpoint

from models import hed_vgg16, dual_hed_vgg16
from dual.combiner import Combiner


def parse_args():
    parser = argparse.ArgumentParser(description='DualNet with HED')

    parser.add_argument('--model', type=str, choices=['hed', 'dual_hed']),
    parser.add_argument('--dual-p', type=float, default=0.25, help='Re-balanced positive sample rate in DualNet')
    parser.add_argument('--dual-stages', type=int, default=2, help='Used stages in DualNet')
    parser.add_argument('--loss', type=str, default='dice', choices=['cbce', 'dice', 'focal'])
    parser.add_argument('--focal-alpha', type=float, default=0.75)
    parser.add_argument('--focal-gamma', type=float, default=2.0)

    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--dataset-test-split', type=str, default='test')
    parser.add_argument('--epochs', default=40, type=int, help='Training epochs')
    parser.add_argument('--pretrained', type=str, default='./dual-HED/models/5stage-vgg.py36pickle',
                        help='Pretrained model path')
    parser.add_argument('--checkpoint', default='', help='Resume from the checkpoint')
    parser.add_argument('--output', default='../output', help='Output path')
    parser.add_argument('--gpus', type=str, default='0', help='CUDA visible devices')
    parser.add_argument('--test-only', default=False, help='Test the model without training', action='store_true')

    parser.add_argument('--batch-size', default=1, type=int, help='Training batch size')
    parser.add_argument('--iter-size', default=1, type=int, help='Training iteration size')
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate')
    parser.add_argument('--lr-step', default=1e9, type=int, help='Learning rate step size')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='Learning rate decay (gamma)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight-decay', default=2e-4, type=float, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=0, help='Num workers')
    parser.add_argument('--print-frequency', default=100, type=int, help='Print frequency by iteration')
    parser.add_argument('--save-interval', type=int, default=1, help='Checkpoint save interval by epoch')
    return parser.parse_args()


args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

board_writer = None
global_step = 0


def main():
    current_dir = abspath(dirname(__file__))
    output_dir = join(current_dir, args.output)
    if not isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not args.test_only:
        log = Logger(join(output_dir, 'train_log.txt'))
        sys.stdout = log

    print(args)

    train_dataset = LesionDataset(name=args.dataset, split='train')
    test_dataset = LesionDataset(name=args.dataset, split=args.dataset_test_split)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              drop_last=True, shuffle=True)
    sub_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers,
                             drop_last=False, shuffle=False)

    if args.model == 'hed':
        net = hed_vgg16.get_model(device, pretrained=args.pretrained)
        opt = hed_vgg16.get_optimizer(net, args.lr, args.momentum, args.weight_decay)
    elif args.model == 'dual_hed':
        net = dual_hed_vgg16.get_model(device, pretrained=args.pretrained)
        opt = dual_hed_vgg16.get_optimizer(net, args.lr, args.momentum, args.weight_decay)

    lr_schd = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)

    if args.checkpoint:
        load_checkpoint(net, opt, args.checkpoint)

    if args.test_only is True:
        test(test_loader, net, save_dir=output_dir, args=args)
    else:
        train_epoch_losses = []
        for epoch in range(args.epochs):
            if not args.model.startswith('dual_'):
                train_epoch_loss = train(train_loader, net, opt, lr_schd, epoch)
            else:
                train_epoch_loss = dual_train(train_loader, sub_train_loader, net, opt, lr_schd, epoch)

            test(test_loader, net, save_dir=join(output_dir, 'epoch-{}-{}'.format(epoch, args.dataset_test_split)))

            log.flush()

            if epoch % args.save_interval == 0:
                save_checkpoint(state={'models': net.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch},
                                path=os.path.join(output_dir, 'epoch-{}-checkpoint.pt'.format(epoch)))
            train_epoch_losses.append(train_epoch_loss)


def dual_train(train_loader, train_loader_sub, net, opt, lr_schd, epoch):
    global global_step
    global board_writer

    net.train()
    opt.zero_grad()
    batch_loss_meter = AverageMeter()
    counter = 0

    combiner = Combiner(net, device, args.epochs, p_rate=args.dual_p)

    sub_iter = iter(train_loader_sub)
    for batch_index, (images, labels) in enumerate(tqdm(train_loader)):
        sub_images, sub_labels = sub_iter.next()

        global_step += 1
        combiner.reset_epoch(epoch)
        combiner.reset_global_iteration(global_step)

        if counter == 0:
            lr_schd.step()
        counter += 1

        images, labels = images.to(device), labels.to(device)
        sub_images, sub_labels = sub_images.to(device), sub_labels.to(device)

        dual_loss = combiner.dual_mix_forward(images, labels, sub_images, sub_labels, stages=args.dual_stages)
        eqv_iter_loss = dual_loss / args.iter_size

        eqv_iter_loss.backward()
        if counter == args.iter_size:
            opt.step()
            opt.zero_grad()
            counter = 0

        batch_loss_meter.update(eqv_iter_loss.item())

        if batch_index % args.print_frequency == args.print_frequency - 1:
            print(('Training epoch:{}/{}, batch:{}/{} current iteration:{}, ' +
                   'current batch batch_loss:{}, epoch average batch_loss:{}, learning rate list:{}').format(
                epoch, args.epochs, batch_index, len(train_loader), lr_schd.last_epoch, batch_loss_meter.val,
                batch_loss_meter.avg, lr_schd.get_lr()))

    return batch_loss_meter.avg


def train(train_loader, net, opt, lr_schd, epoch):
    global global_step

    net.train()
    opt.zero_grad()
    batch_loss_meter = AverageMeter()
    counter = 0

    for batch_index, (images, labels) in enumerate(tqdm(train_loader)):
        global_step += 1

        if counter == 0:
            lr_schd.step()
        counter += 1

        images, labels = images.to(device), labels.to(device)

        preds_list = net(images)
        batch_loss = sum([get_loss(preds, labels) for preds in preds_list])
        eqv_iter_loss = batch_loss / args.iter_size

        eqv_iter_loss.backward()
        if counter == args.iter_size:
            opt.step()
            opt.zero_grad()
            counter = 0

        batch_loss_meter.update(batch_loss.item())

        if batch_index % args.print_frequency == args.print_frequency - 1:
            print(('Training epoch:{}/{}, batch:{}/{} current iteration:{}, ' +
                   'current batch batch_loss:{}, epoch average batch_loss:{}, learning rate list:{}'
                   ).format(epoch, args.epochs, batch_index, len(train_loader), lr_schd.last_epoch,
                            batch_loss_meter.val, batch_loss_meter.avg, lr_schd.get_lr()))

    return batch_loss_meter.avg


def test(test_loader, net, save_dir, args=None):
    if not isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    net.eval()

    for batch_index, images in enumerate(tqdm(test_loader)):
        images = images.cuda()
        _, _, h, w = images.shape
        preds_list = net(images)
        fuse = preds_list[-1].detach().cpu().numpy()[0, 0]
        name = test_loader.dataset.images_name[batch_index]
        Image.fromarray((fuse * 255).astype(np.uint8)).save(join(save_dir, '{}.png'.format(name)))

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()


def get_loss(preds, labels):
    mask = (labels > 0.5).float()
    b, c, h, w = mask.shape

    if args.loss == 'cbce':
        return weighted_cross_entropy_loss(preds, labels)
    elif args.loss == 'dice':
        losses = 1. - dice_coef(labels.float(), preds.float())
    elif args.loss == 'focal':
        losses = focal_loss_with_logits(preds.float(), mask, alpha=args.focal_alpha, gamma=args.focal_gamma)
        return losses
    else:
        losses = torch.nn.functional.binary_cross_entropy(preds.float(), labels.float(), reduction='none')

    loss = torch.sum(losses) / b
    return loss


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    return 2 * (intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def weighted_cross_entropy_loss(preds, labels):
    mask = (labels > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()
    num_neg = c * h * w - num_pos
    weight = torch.zeros_like(mask)
    weight[labels > 0.5] = num_neg / (num_pos + num_neg)
    weight[labels <= 0.5] = num_pos / (num_pos + num_neg)
    losses = torch.nn.functional.binary_cross_entropy(preds.float(), labels.float(), weight=weight, reduction='none')
    loss = torch.sum(losses) / b
    return loss


def focal_loss_with_logits(output,
                           target,
                           gamma=2.0,
                           alpha=0.25,
                           reduction="mean",
                           normalized=False,
                           reduced_threshold=None,
                           eps=1e-6):
    target = target.type(output.type())

    output = output.view(-1)
    target = target.view(-1)

    logpt = F.binary_cross_entropy(output, target, reduction="none")
    pt = torch.exp(-logpt)

    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


if __name__ == '__main__':
    main()
