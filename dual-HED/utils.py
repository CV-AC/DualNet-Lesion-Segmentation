import os
import sys
import torch


class Logger(object):
    def __init__(self, path=None):
        self.console = sys.stdout
        self.file = None
        if path is not None:
            self.file = open(path, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def save_checkpoint(state, path='./checkpoint.pth'):
    torch.save(state, path)


def load_checkpoint(net, opt, path='./checkpoint.pth'):
    if os.path.isfile(path):
        print('=> Loading checkpoint {}...'.format(path))
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['models'])
        opt.load_state_dict(checkpoint['opt'])
        return checkpoint['epoch']
    else:
        raise ValueError('=> No checkpoint found at {}.'.format(path))
