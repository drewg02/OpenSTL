import argparse
import pickle
import torch
import os

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
    # parser.add_argument('--test', action='store_true', default=False, help='Only performs testing')
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
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize the results when testing')

    parser.add_argument('--image_height', type=int, default=64)
    parser.add_argument('--image_width', type=int, default=64)
    parser.add_argument('--datafile', type=str, default='data/2dplate/dataset.pkl')

    return parser


def create_dataloaders(dataset, batch_size):
    dataloader_train, dataloader_val = None, None
    if 'X_train' in dataset and 'X_val' in dataset:
        x_train, X_val, X_test = dataset['X_train'], dataset['X_val'], dataset['X_test']
        Y_train, Y_val, Y_test = dataset['Y_train'], dataset['Y_val'], dataset['Y_test']

        train_set = SimulationDataset(X=X_train, Y=Y_train)
        val_set = SimulationDataset(X=X_val, Y=Y_val)
        test_set = SimulationDataset(X=X_test, Y=Y_test)

        dataloader_train = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        dataloader_val = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    else:
        X_test, Y_test = dataset['X'], dataset['Y']
        test_set = SimulationDataset(X=X_test, Y=Y_test)

    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    if dataloader_train and dataloader_val:
        return dataloader_train, dataloader_val, dataloader_test
    else:
        return dataloader_test


def generate_configs(ex_name, pre_seq_length, aft_seq_length, batch_size, args):
    custom_training_config = {
        'pre_seq_length': pre_seq_length,
        'aft_seq_length': aft_seq_length,
        'total_length': pre_seq_length + aft_seq_length,
        'batch_size': batch_size,
        'val_batch_size': batch_size,
        'epoch': args.epoch,
        'lr': args.lr,
        'metrics': ['mse', 'mae'],

        'ex_name': ex_name,
        'dataname': '2dplate',
        'in_shape': [pre_seq_length, 1, args.image_height, args.image_width],
    }

    custom_model_config = {
        'method': 'E3DLSTM',
        # reverse scheduled sampling
        'reverse_scheduled_sampling': 0,
        'r_sampling_step_1': 25000,
        'r_sampling_step_2': 50000,
        'r_exp_alpha': 5000,
        # scheduled sampling
        'scheduled_sampling': 1,
        'sampling_stop_iter': 50000,
        'sampling_start_value': 1.0,
        'sampling_changing_rate': 0.00002,
        # model
        'num_hidden': '128,128,128,128',
        'filter_size': 5,
        'stride': 1,
        'patch_size': 4,
        'layer_norm': 0,
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



def save_data(dataset, file_path):
    folder_path = os.path.dirname(file_path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)


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