import argparse
import json
import os
import pickle
import numpy as np

from openstl.simulation import SimulationDataset
from openstl.simulation.simulations import simulations
from openstl.datasets.utils import create_loader


def create_parser():
    parser = argparse.ArgumentParser(
        description='OpenSTL train/test a model')
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--dist', action='store_true', default=False,
                        help='Whether to use distributed training (DDP)')
    parser.add_argument('--display_step', default=10, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='work_dirs', type=str)
    parser.add_argument('--ex_name', '-ex', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Whether to use Native AMP for mixed precision training (PyTorch=>1.6.0)')
    parser.add_argument('--torchscript', action='store_true', default=False,
                        help='Whether to use torchscripted model')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--diff_seed', action='store_true', default=False,
                        help='Whether to set different seeds for different ranks')
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to measure inference speed (FPS)')
    parser.add_argument('--empty_cache', action='store_true', default=True,
                        help='Whether to empty cuda cache after GPU training')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False,
                        help='Whether to find unused parameters in forward during DDP training')
    parser.add_argument('--broadcast_buffers', action='store_false', default=True,
                        help='Whether to set broadcast_buffers to false during DDP training')
    parser.add_argument('--resume_from', type=str, default=None, help='the checkpoint file to resume from')
    parser.add_argument('--auto_resume', action='store_true', default=False,
                        help='When training was interupted, resume from the latest checkpoint')
    parser.add_argument('--test', action='store_true', default=False, help='Perform testing')
    parser.add_argument('--inference', '-i', action='store_true', default=False, help='Only performs inference')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='whether to set deterministic options for CUDNN backend (reproducable)')
    parser.add_argument('--launcher', default='none', type=str,
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        help='job launcher for distributed training')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
                        help='port only works when launcher=="slurm"')

    # dataset parameters
    parser.add_argument('--batch_size', '-b', default=16, type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', '-vb', default=16, type=int, help='Validation batch size')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--dataname', '-d', default='mmnist', type=str,
                        help='Dataset name (default: "mmnist")')
    parser.add_argument('--pre_seq_length', default=None, type=int, help='Sequence length before prediction')
    parser.add_argument('--aft_seq_length', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--total_length', default=None, type=int, help='Total Sequence length for prediction')
    parser.add_argument('--use_augment', action='store_true', default=False,
                        help='Whether to use image augmentations for training')
    parser.add_argument('--use_prefetcher', action='store_true', default=False,
                        help='Whether to use prefetcher for faster data loading')
    parser.add_argument('--drop_last', action='store_true', default=False,
                        help='Whether to drop the last batch in the val data loading')

    # method parameters
    parser.add_argument('--method', '-m', default='SimVP', type=str,
                        choices=['ConvLSTM', 'convlstm', 'CrevNet', 'crevnet', 'DMVFN', 'dmvfn', 'E3DLSTM', 'e3dlstm',
                                 'MAU', 'mau', 'MIM', 'mim', 'PhyDNet', 'phydnet', 'PredNet', 'prednet',
                                 'PredRNN', 'predrnn', 'PredRNNpp', 'predrnnpp', 'PredRNNv2', 'predrnnv2',
                                 'SimVP', 'simvp', 'TAU', 'tau'],
                        help='Name of video prediction method to train (default: "SimVP")')
    parser.add_argument('--config_file', '-c', default=None, type=str,
                        help='Path to the default config file')
    parser.add_argument('--model_type', default=None, type=str,
                        help='Name of model for SimVP (default: None)')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate(default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate for SimVP (default: 0.)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Whether to allow overwriting the provided config file with args')

    # Training parameters (optimizer)
    parser.add_argument('--epoch', '-e', default=None, type=int, help='end epochs (default: 200)')
    parser.add_argument('--checkpoint_interval', '-ci', default=None, type=int,
                        help='Checkpoint save interval (default: None)')
    parser.add_argument('--log_step', default=1, type=int, help='Log interval by step')
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=0., type=float, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--early_stop_epoch', default=-1, type=int,
                        help='Check to early stop after this epoch')
    parser.add_argument('--no_display_method_info', action='store_true', default=False,
                        help='Do not display method info')

    # Training parameters (scheduler)
    parser.add_argument('--sched', default=None, type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--lr', default=None, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_epoch', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='Whether to set the weight decay of bias and bn to 0')

    # Simulation parameters
    parser.add_argument('--datafile_in', type=str, required=True,
                        help='Specifies the input data file path.')

    return parser


def create_dataloader(data, pre_seq_length=10, aft_seq_length=10, batch_size=16, shuffle=False, is_training=False, distributed=False):
    dataset = SimulationDataset(data, pre_seq_length, aft_seq_length)

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

    train_loader = create_dataloader(loader['train'], pre_seq_length, aft_seq_length, batch_size, True, True, distributed)
    val_loader = create_dataloader(loader['validation'], pre_seq_length, aft_seq_length, val_batch_size, False, False, distributed)
    test_loader = create_dataloader(loader['test'], pre_seq_length, aft_seq_length, test_batch_size, False, False, distributed)

    return train_loader, val_loader, test_loader


def generate_config(args, metrics=None):
    if not metrics:
        metrics = ['mse', 'mae', 'ssim']

    pre_seq_length = args.pre_seq_length
    aft_seq_length = args.aft_seq_length

    with open(args.datafile_in, 'r') as f:
        data = json.load(f)

        key = 'train' if data['train'] else 'test'

        example_sample = np.load(data[key]['samples'][0][0])
        sample_shape = example_sample.shape

        channels = len(sample_shape) > 2 and sample_shape[0] or 1
        image_height = sample_shape[-2]
        image_width = sample_shape[-1]
    
    training_config = {
        'pre_seq_length': pre_seq_length,
        'aft_seq_length': aft_seq_length,
        'total_length': pre_seq_length + aft_seq_length,
        'batch_size': args.batch_size,
        'val_batch_size': args.val_batch_size,
        'epoch': args.epoch,
        'lr': args.lr,
        'metrics': metrics,

        'ex_name': args.ex_name,
        'dataname': 'simulation',
        'in_shape': [pre_seq_length, channels, image_height, image_width],
    }

    return training_config


def get_simulation_class(simulation_name):
    simulation_class = [simulation for simulation in simulations if
                        simulation.__name__.lower() == simulation_name.lower()]
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {simulation_name}")
    return simulation_class[0]


def load_data(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset


def save_data(dataset, file_path):
    folder_path = os.path.dirname(file_path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
