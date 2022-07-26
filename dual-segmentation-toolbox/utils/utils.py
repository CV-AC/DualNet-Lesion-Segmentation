import time
from collections import OrderedDict
from .logger import get_logger
import torch
import os

logger = get_logger()


def load_dual_model(model, model_file, mapping_dict=None, logging_keys=True):
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        state_dict = torch.load(model_file, map_location=device)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    prefixs = ['backbone.', 'cb_block.', 'rb_block.']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if mapping_dict and (k in mapping_dict):
            names = mapping_dict[k]
            for name in names:
                new_state_dict[name] = v
        else:
            for prefix in prefixs:
                name = prefix + k
                new_state_dict[name] = v
    state_dict = new_state_dict

    print()

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if logging_keys:
        if len(missing_keys) > 0:
            logger.warning('Missing key(s) in state_dict: {}'.format(
                ', '.join('{}'.format(k) for k in missing_keys)))

        if len(unexpected_keys) > 0:
            logger.warning('Unexpected key(s) in state_dict: {}'.format(
                ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model


def save_checkpoint(model, opt, global_iteration, path):
    torch.save({
        'global_iteration': global_iteration,
        'model_state_dict': model.state_dict(),
        'opt': opt.state_dict()}, path)


def load_checkpoint(model, opt, path):
    if os.path.isfile(path):
        print('=> Loading checkpoint {}...'.format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['opt'])
        return checkpoint['global_iteration']
    else:
        raise ValueError('=> No checkpoint found at {}.'.format(path))


def load_test_checkpoint(model, path):
    if os.path.isfile(path):
        print('=> Loading checkpoint {}...'.format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError('=> No checkpoint found at {}.'.format(path))
