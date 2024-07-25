import json
import os
import random
from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial
from itertools import repeat
from typing import Callable

import numpy as np
import torch
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler

from .simvp_dataset import SimVP_Dataset


def create_parser():
    parser = ArgumentParser(description='SimVP training')
    parser.add_argument('--datafile_in', type=str, default=None, help='Path to the loader file')
    parser.add_argument('--config_file', type=str, default=None, help='Path to the config file')
    parser.add_argument('--ex_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--work_dirs', type=str, default='./work_dirs', help='Working directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--spatio_kernel_enc', type=int, default=3, help='Spatial kernel size for encoder')
    parser.add_argument('--spatio_kernel_dec', type=int, default=3, help='Spatial kernel size for decoder')
    parser.add_argument('--hid_S', type=int, default=64, help='Hidden size for spatial encoder')
    parser.add_argument('--hid_T', type=int, default=512, help='Hidden size for temporal encoder')
    parser.add_argument('--N_T', type=int, default=8, help='Number of temporal layers')
    parser.add_argument('--N_S', type=int, default=4, help='Number of spatial layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--val_batch_size', type=int, default=16, help='Validation batch size')
    parser.add_argument('--drop_path', type=float, default=0, help='Drop path rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--use_gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--dist', action='store_true', help='Use distributed training')
    parser.add_argument('--pre_seq_length', type=int, default=10, help='Length of the input sequence')
    parser.add_argument('--aft_seq_length', type=int, default=10, help='Length of the output sequence')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--inference', action='store_true', help='Test the model')
    parser.add_argument('--clip_grad', type=float, default=0, help='Gradient clipping value')
    parser.add_argument('--clip_mode', type=str, default='norm', help='Gradient clipping mode')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--opt_eps', type=float, default=None, help='Optimizer epsilon')
    parser.add_argument('--opt_betas', type=float, nargs='+', default=None, help='Optimizer betas')
    parser.add_argument('--empty_cache', default=True, action='store_true', help='Empty cache')

    return parser


seconds_format_dict = {
    'years': 31536000,
    'months': 2592000,
    'weeks': 604800,
    'days': 86400,
    'hours': 3600,
    'minutes': 60,
    'seconds': 1
}


def format_seconds(seconds):
    time_str = ''
    for key, value in seconds_format_dict.items():
        if seconds >= value:
            time_str += f'{seconds // value} {key}, '
            seconds = seconds % value

    return time_str[:-2]


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class Recorder:
    def __init__(self, verbose=False, delta=0, early_stop_time=10):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.decrease_time = 0
        self.early_stop_time = early_stop_time

    def __call__(self, val_loss, model, path, early_stop=False):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.decrease_time = 0
        else:
            self.decrease_time += 1
        # return self.decrease_time <= self.early_stop_time if early_stop else 0
        return True if early_stop else 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def _weights_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(  # type: ignore
        state_dict, '_metadata', OrderedDict())
    return state_dict_cpu


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x


def worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding
        # is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.float()
                        next_target = next_target.float()

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(dataset,
                  batch_size,
                  shuffle=True,
                  is_training=False,
                  mean=None,
                  std=None,
                  num_workers=1,
                  num_aug_repeats=0,
                  input_channels=1,
                  use_prefetcher=False,
                  distributed=False,
                  pin_memory=False,
                  drop_last=False,
                  fp16=False,
                  collate_fn=None,
                  persistent_workers=True,
                  worker_seeding='all'):
    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment is not supported in non-distributed or IterableDataset"

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle and (
            not isinstance(dataset, torch.utils.data.IterableDataset)) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_channels,
            fp16=fp16,
        )

    return loader


def create_dataloader(data, pre_seq_length=10, aft_seq_length=10, batch_size=16, shuffle=False, is_training=False,
                      distributed=False):
    dataset = SimVP_Dataset(data, pre_seq_length, aft_seq_length)

    dataloader = create_loader(dataset, batch_size,
                               shuffle=shuffle, is_training=is_training,
                               distributed=distributed, num_workers=4)

    return dataloader


def create_dataloaders(file_path_or_data, pre_seq_length=10, aft_seq_length=10, batch_size=16, val_batch_size=4,
                       test_batch_size=4, distributed=False):
    if type(file_path_or_data) == str:
        if not os.path.exists(file_path_or_data):
            return None, None, None

        with open(file_path_or_data, 'r') as f:
            loader = json.load(f)
    else:
        loader = file_path_or_data

    train_loader, val_loader, test_loader = None, None, None
    if 'train' in loader:
        train_loader = create_dataloader(loader['train'], pre_seq_length, aft_seq_length, batch_size, True, True,
                                         distributed)
    if 'validation' in loader:
        val_loader = create_dataloader(loader['validation'], pre_seq_length, aft_seq_length, val_batch_size, False,
                                       False, distributed)
    if 'test' in loader:
        test_loader = create_dataloader(loader['test'], pre_seq_length, aft_seq_length, test_batch_size, False, False,
                                        distributed)

    return train_loader, val_loader, test_loader


def measure_throughput(model, input_dummy):
    def get_batch_size(H, W):
        max_side = max(H, W)
        if max_side >= 128:
            bs = 10
            repetitions = 1000
        else:
            bs = 100
            repetitions = 100
        return bs, repetitions

    if isinstance(input_dummy, tuple):
        input_dummy = list(input_dummy)
        _, T, C, H, W = input_dummy[0].shape
        bs, repetitions = get_batch_size(H, W)
        _input = torch.rand(bs, T, C, H, W).to(input_dummy[0].device)
        input_dummy[0] = _input
        input_dummy = tuple(input_dummy)
    else:
        _, T, C, H, W = input_dummy.shape
        bs, repetitions = get_batch_size(H, W)
        input_dummy = torch.rand(bs, T, C, H, W).to(input_dummy.device)
    total_time = 0
    with torch.no_grad():
        for _ in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            if isinstance(input_dummy, tuple):
                _ = model(*input_dummy)
            else:
                _ = model(input_dummy)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * bs) / total_time
    return Throughput
