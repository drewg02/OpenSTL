import argparse
import pickle
import os
import json

from torch.utils.data import DataLoader

from openstl.simulation import SimulationDataset
from openstl.simulation.simulations import simulations

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
    parser.add_argument('--local_rank', type=int, default=0)
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
    parser.add_argument('--config_file', '-c', default='configs/mmnist/simvp/SimVP_gSTA.py', type=str,
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
    parser.add_argument('--train', action='store_true', default=False, help='Perform training')
    parser.add_argument('--datafolder_in', type=str, required=True,
                        help='Specifies the input data file path.')

    return parser

def create_data_loader(datafolder_in, file_name, pre_seq_length=10, aft_seq_length=10, batch_size=16, shuffle=False):
    file_path = os.path.join(datafolder_in, file_name)

    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        file_paths = json.load(f)

    if not file_paths:
        return None

    dataset = SimulationDataset(file_paths, pre_seq_length, aft_seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_data_loaders(datafolder_in, pre_seq_length=10, aft_seq_length=10, batch_size=16, val_batch_size=16, test_batch_size=16):
    train_loader = create_data_loader(datafolder_in, 'train_files.json', pre_seq_length, aft_seq_length, batch_size, True)
    val_loader = create_data_loader(datafolder_in, 'val_files.json', pre_seq_length, aft_seq_length, val_batch_size, True)
    test_loader = create_data_loader(datafolder_in, 'test_files.json', pre_seq_length, aft_seq_length, test_batch_size, True)

    return train_loader, val_loader, test_loader


def generate_configs(pre_seq_length, aft_seq_length, image_height, image_width, args):
    custom_training_config = {
        'pre_seq_length': pre_seq_length,
        'aft_seq_length': aft_seq_length,
        'total_length': pre_seq_length + aft_seq_length,
        'batch_size': args.batch_size,
        'val_batch_size': args.batch_size,
        'epoch': args.epoch,
        'lr': args.lr,
        'metrics': ['mse', 'mae', 'ssim'],

        'ex_name': args.ex_name,
        'dataname': '2dplate',
        'in_shape': [pre_seq_length, 1, image_height, image_width],
    }

    custom_model_config = {
        'method': 'SimVP',
        # model
        'spatio_kernel_enc': 3,
        'spatio_kernel_dec': 3,
        # model_type = None  # define `model_type` in args
        'hid_S': 64,
        'hid_T': 512,
        'N_T': 8,
        'N_S': 4,
        # training
        'lr': 1e-3,
        'batch_size': 16,
        'drop_path': 0,
        'sched': 'onecycle'
    }

    return custom_training_config, custom_model_config

def get_simulation_class(simulation_name):
    simulation_class = [simulation for simulation in simulations if simulation.__name__ == simulation_name]
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {simulation_name}")
    return simulation_class[0]

def get_seq_lengths(args):
    pre_seq_length = args.pre_seq_length
    aft_seq_length =  args.aft_seq_length
    total_length = args.total_length
    if pre_seq_length and aft_seq_length and not total_length:
        total_length = pre_seq_length + aft_seq_length

    if pre_seq_length and aft_seq_length and pre_seq_length + aft_seq_length != total_length:
        raise ValueError(
            f"pre_seq_length ({pre_seq_length}) + aft_seq_length ({aft_seq_length}) must be equal to total_length ({total_length})")

    return pre_seq_length, aft_seq_length, total_length

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