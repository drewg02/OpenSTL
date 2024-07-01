import re
import pandas as pd
from datetime import timedelta
import os
import argparse

hostname_re = re.compile(r'Running on hostname:\s*(.*)')
experiment_re = re.compile(r'---\[ Experiment: (.*) \]---')
world_size_re = re.compile(r'Distributed world_size=(\d+)')
rank_pattern_re = re.compile(r'Use distributed mode with GPUs: local rank=\d+')
config_re = re.compile(r'loading config from (.*) \.\.\.')
device_re = re.compile(r'device:\s*(.*)')
dist_re = re.compile(r'dist:\s*(True|False)')
display_step_re = re.compile(r'display_step:\s*(\d+)')
res_dir_re = re.compile(r'res_dir:\s*(.*)')
ex_name_re = re.compile(r'ex_name:\s*(.*)')
use_gpu_re = re.compile(r'use_gpu:\s*(True|False)')
fp16_re = re.compile(r'fp16:\s*(True|False)')
torchscript_re = re.compile(r'torchscript:\s*(True|False)')
seed_re = re.compile(r'seed:\s*(\d+)')
diff_seed_re = re.compile(r'diff_seed:\s*(True|False)')
fps_re = re.compile(r'fps:\s*(True|False)')
empty_cache_re = re.compile(r'empty_cache:\s*(True|False)')
find_unused_parameters_re = re.compile(r'find_unused_parameters:\s*(True|False)')
broadcast_buffers_re = re.compile(r'broadcast_buffers:\s*(True|False)')
resume_from_re = re.compile(r'resume_from:\s*(.*)')
auto_resume_re = re.compile(r'auto_resume:\s*(True|False)')
test_re = re.compile(r'test:\s*(True|False)')
inference_re = re.compile(r'inference:\s*(True|False)')
deterministic_re = re.compile(r'deterministic:\s*(True|False)')
launcher_re = re.compile(r'launcher:\s*(.*)')
local_rank_re = re.compile(r'local_rank:\s*(\d+)')
port_re = re.compile(r'port:\s*(\d+)')
batch_size_re = re.compile(r'batch_size:\s*(\d+)')
val_batch_size_re = re.compile(r'val_batch_size:\s*(\d+)')
num_workers_re = re.compile(r'num_workers:\s*(\d+)')
data_root_re = re.compile(r'data_root:\s*(.*)')
dataname_re = re.compile(r'dataname:\s*(.*)')
pre_seq_length_re = re.compile(r'pre_seq_length:\s*(\d+)')
aft_seq_length_re = re.compile(r'aft_seq_length:\s*(\d+)')
total_length_re = re.compile(r'total_length:\s*(\d+)')
use_augment_re = re.compile(r'use_augment:\s*(True|False)')
use_prefetcher_re = re.compile(r'use_prefetcher:\s*(True|False)')
drop_last_re = re.compile(r'drop_last:\s*(True|False)')
method_re = re.compile(r'method:\s*(.*)')
config_file_re = re.compile(r'config_file:\s*(.*)')
model_type_re = re.compile(r'model_type:\s*(.*)')
drop_re = re.compile(r'drop:\s*(\d+\.\d+)')
drop_path_re = re.compile(r'drop_path:\s*(\d+)')
overwrite_re = re.compile(r'overwrite:\s*(True|False)')
epoch_re = re.compile(r'epoch:\s*(\d+)')
checkpoint_interval_re = re.compile(r'checkpoint_interval:\s*(.*)')
log_step_re = re.compile(r'log_step:\s*(\d+)')
opt_re = re.compile(r'opt:\s*(.*)')
opt_eps_re = re.compile(r'opt_eps:\s*(.*)')
opt_betas_re = re.compile(r'opt_betas:\s*(.*)')
momentum_re = re.compile(r'momentum:\s*(\d+\.\d+)')
weight_decay_re = re.compile(r'weight_decay:\s*(\d+\.\d+)')
clip_grad_re = re.compile(r'clip_grad:\s*(.*)')
clip_mode_re = re.compile(r'clip_mode:\s*(.*)')
early_stop_epoch_re = re.compile(r'early_stop_epoch:\s*(.*)')
no_display_method_info_re = re.compile(r'no_display_method_info:\s*(True|False)')
sched_re = re.compile(r'sched:\s*(.*)')
lr_re = re.compile(r'lr:\s*(\d+\.\d+)')
lr_k_decay_re = re.compile(r'lr_k_decay:\s*(\d+\.\d+)')
warmup_lr_re = re.compile(r'warmup_lr:\s*(\d+e-\d+)')
min_lr_re = re.compile(r'min_lr:\s*(\d+e-\d+)')
final_div_factor_re = re.compile(r'final_div_factor:\s*(\d+\.\d+)')
warmup_epoch_re = re.compile(r'warmup_epoch:\s*(\d+)')
decay_epoch_re = re.compile(r'decay_epoch:\s*(\d+)')
decay_rate_re = re.compile(r'decay_rate:\s*(\d+\.\d+)')
filter_bias_and_bn_re = re.compile(r'filter_bias_and_bn:\s*(True|False)')
datafile_in_re = re.compile(r'datafile_in:\s*(.*)')
saved_path_re = re.compile(r'saved_path:\s*(.*)')
metrics_re = re.compile(r'metrics:\s*\[(.*)\]')
in_shape_re = re.compile(r'in_shape:\s*\[(.*)\]')
spatio_kernel_enc_re = re.compile(r'spatio_kernel_enc:\s*(\d+)')
spatio_kernel_dec_re = re.compile(r'spatio_kernel_dec:\s*(\d+)')
hid_S_re = re.compile(r'hid_S:\s*(\d+)')
hid_T_re = re.compile(r'hid_T:\s*(\d+)')
N_T_re = re.compile(r'N_T:\s*(\d+)')
N_S_re = re.compile(r'N_S:\s*(\d+)')

training_info_re = re.compile(
    r'Epoch: (\d+), Steps: (\d+) \| Lr: ([\d.]+) \| Train Loss: ([\d.]+) \| Vali Loss: ([\d.]+)')
training_length_re = re.compile(r'100%\|██████████\| \d+/\d+ \[(\d+):(\d+)<\d+:\d+, *\d+\.\d+(it/s|s/it)\]\n\[')
validation_metrics_re = re.compile(
    r'\[>{7,}\] \d+/\d+, [\d.]+ task/s, elapsed: (\d+)s, ETA:.*?mse:([\d.]+), mae:([\d.]+), ssim:([\d.]+)',
    re.MULTILINE)
training_time_re = re.compile(r'Training time: (\d+) days, (\d+):(\d+):(\d+)')


def parse_log(log_file):
    # Extract job_id from filename
    job_id = os.path.basename(log_file).split('-')[1].split('.')[0]

    with open(log_file, 'r') as file:
        log_content = file.read()

    rank_matches = re.findall(rank_pattern_re, log_content)
    num_gpus = len(rank_matches)

    hostname_match = hostname_re.search(log_content)
    experiment_match = experiment_re.search(log_content)
    world_size_match = world_size_re.search(log_content)
    config_file_match = config_re.search(log_content)
    ex_name_match = ex_name_re.search(log_content)
    seed_match = seed_re.search(log_content)
    batch_size_match = batch_size_re.search(log_content)
    val_batch_size_match = val_batch_size_re.search(log_content)
    pre_seq_length_match = pre_seq_length_re.search(log_content)
    aft_seq_length_match = aft_seq_length_re.search(log_content)
    method_match = method_re.search(log_content)
    epoch_match = epoch_re.search(log_content)
    in_shape_match = in_shape_re.search(log_content)

    if not hostname_match:
        print("Hostname not found in log file.")
    if not experiment_match:
        print("Experiment not found in log file.")
    if not world_size_match:
        print("World size not found in log file.")
    if not config_file_match:
        print("Config file not found in log file.")
    if not ex_name_match:
        print("Experiment name (ex_name) not found in log file.")
    if not seed_match:
        print("Seed not found in log file.")
    if not batch_size_match:
        print("Batch size not found in log file.")
    if not val_batch_size_match:
        print("Validation batch size not found in log file.")
    if not pre_seq_length_match:
        print("Pre-sequence length not found in log file.")
    if not aft_seq_length_match:
        print("After-sequence length not found in log file.")
    if not method_match:
        print("Method not found in log file.")
    if not epoch_match:
        print("Epoch not found in log file.")
    if not in_shape_match:
        print("Input shape not found in log file.")

    if not hostname_match or not experiment_match or not config_file_match or not ex_name_match or not seed_match or not batch_size_match or not val_batch_size_match or not pre_seq_length_match or not aft_seq_length_match or not method_match or not config_file_match or not epoch_match or not in_shape_match:
        raise ValueError("Log file does not contain the necessary information.")

    hostname = hostname_match.group(1)
    experiment = experiment_match.group(1)
    world_size = world_size_match.group(1) if world_size_match else 1

    training_epochs = {}
    training_metrics_matches = list(re.finditer(training_info_re, log_content))
    training_length_matches = list(re.finditer(training_length_re, log_content))
    validation_metrics_matches = list(re.finditer(validation_metrics_re, log_content))

    if len(training_metrics_matches) != len(training_length_matches) != len(validation_metrics_matches) - 1:
        print(len(training_metrics_matches))
        print(len(training_length_matches))
        print(len(validation_metrics_matches))
        print("Mismatch between the number of training and validation metric matches.")
        raise ValueError("Log file does not contain matching training and validation information.")

    for train_match, train_length_match, val_match in zip(training_metrics_matches, training_length_matches,
                                                          validation_metrics_matches[:-1]):
        epoch_num = int(train_match.group(1))
        steps = int(train_match.group(2))
        lr = float(train_match.group(3))
        train_loss = float(train_match.group(4))
        vali_loss = float(train_match.group(5))

        elapsed = int(val_match.group(1))
        mse = float(val_match.group(2))
        mae = float(val_match.group(3))
        ssim = float(val_match.group(4))

        minutes = int(train_length_match.group(1))
        seconds = int(train_length_match.group(2))
        total_seconds = minutes * 60 + seconds

        training_epochs[epoch_num] = {
            'steps': steps,
            'lr': lr,
            'train_loss': train_loss,
            'vali_loss': vali_loss,
            'train_seconds': total_seconds,
            'val_seconds': elapsed,
            'mse': mse,
            'mae': mae,
            'ssim': ssim
        }

    training_time_match = training_time_re.search(log_content)
    training_time_seconds = int(timedelta(days=int(training_time_match.group(1)),
                                          hours=int(training_time_match.group(2)),
                                          minutes=int(training_time_match.group(3)),
                                          seconds=int(training_time_match.group(4))).total_seconds())

    final_test_match = validation_metrics_matches[-1]
    test_time = int(final_test_match.group(1))
    test_mse = float(final_test_match.group(2))
    test_mae = float(final_test_match.group(3))
    test_ssim = float(final_test_match.group(4))

    csv_line = {
        'jobid': job_id,
        'hostname': hostname,
        'experiment': experiment,
        'world_size': world_size,
        'training_time_seconds': training_time_seconds,
        'test_time': test_time,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_ssim': test_ssim,
        'num_gpus': num_gpus,
        'device': device_re.search(log_content).group(1),
        'dist': dist_re.search(log_content).group(1) == 'True',
        'display_step': int(display_step_re.search(log_content).group(1)),
        'res_dir': res_dir_re.search(log_content).group(1),
        'ex_name': ex_name_re.search(log_content).group(1),
        'use_gpu': use_gpu_re.search(log_content).group(1) == 'True',
        'fp16': fp16_re.search(log_content).group(1) == 'False',
        'torchscript': torchscript_re.search(log_content).group(1) == 'False',
        'seed': int(seed_re.search(log_content).group(1)),
        'diff_seed': diff_seed_re.search(log_content).group(1) == 'False',
        'fps': fps_re.search(log_content).group(1) == 'False',
        'empty_cache': empty_cache_re.search(log_content).group(1) == 'True',
        'find_unused_parameters': find_unused_parameters_re.search(log_content).group(1) == 'False',
        'broadcast_buffers': broadcast_buffers_re.search(log_content).group(1) == 'True',
        'resume_from': resume_from_re.search(log_content).group(1),
        'auto_resume': auto_resume_re.search(log_content).group(1) == 'False',
        'test': test_re.search(log_content).group(1) == 'False',
        'inference': inference_re.search(log_content).group(1) == 'False',
        'deterministic': deterministic_re.search(log_content).group(1) == 'False',
        'launcher': launcher_re.search(log_content).group(1),
        'local_rank': int(local_rank_re.search(log_content).group(1)),
        'port': int(port_re.search(log_content).group(1)),
        'batch_size': int(batch_size_re.search(log_content).group(1)),
        'val_batch_size': int(val_batch_size_re.search(log_content).group(1)),
        'num_workers': int(num_workers_re.search(log_content).group(1)),
        'data_root': data_root_re.search(log_content).group(1),
        'dataname': dataname_re.search(log_content).group(1),
        'pre_seq_length': int(pre_seq_length_re.search(log_content).group(1)),
        'aft_seq_length': int(aft_seq_length_re.search(log_content).group(1)),
        'total_length': int(total_length_re.search(log_content).group(1)),
        'use_augment': use_augment_re.search(log_content).group(1) == 'False',
        'use_prefetcher': use_prefetcher_re.search(log_content).group(1) == 'False',
        'drop_last': drop_last_re.search(log_content).group(1) == 'False',
        'method': method_re.search(log_content).group(1),
        'config_file': config_file_re.search(log_content).group(1),
        'model_type': model_type_re.search(log_content).group(1),
        'drop': float(drop_re.search(log_content).group(1)),
        'drop_path': int(drop_path_re.search(log_content).group(1)),
        'overwrite': overwrite_re.search(log_content).group(1) == 'False',
        'epoch': int(epoch_re.search(log_content).group(1)),
        'checkpoint_interval': checkpoint_interval_re.search(log_content).group(1),
        'log_step': int(log_step_re.search(log_content).group(1)),
        'opt': opt_re.search(log_content).group(1),
        'opt_eps': opt_eps_re.search(log_content).group(1),
        'opt_betas': opt_betas_re.search(log_content).group(1),
        'momentum': float(momentum_re.search(log_content).group(1)),
        'weight_decay': float(weight_decay_re.search(log_content).group(1)),
        'clip_grad': clip_grad_re.search(log_content).group(1),
        'clip_mode': clip_mode_re.search(log_content).group(1),
        'early_stop_epoch': early_stop_epoch_re.search(log_content).group(1),
        'no_display_method_info': no_display_method_info_re.search(log_content).group(1) == 'False',
        'sched': sched_re.search(log_content).group(1),
        'lr': float(lr_re.search(log_content).group(1)),
        'lr_k_decay': float(lr_k_decay_re.search(log_content).group(1)),
        'warmup_lr': float(warmup_lr_re.search(log_content).group(1)),
        'min_lr': float(min_lr_re.search(log_content).group(1)),
        'final_div_factor': float(final_div_factor_re.search(log_content).group(1)),
        'warmup_epoch': int(warmup_epoch_re.search(log_content).group(1)),
        'decay_epoch': int(decay_epoch_re.search(log_content).group(1)),
        'decay_rate': float(decay_rate_re.search(log_content).group(1)),
        'filter_bias_and_bn': filter_bias_and_bn_re.search(log_content).group(1) == 'False',
        'datafile_in': datafile_in_re.search(log_content).group(1),
        'saved_path': saved_path_re.search(log_content).group(1),
        'metrics': metrics_re.search(log_content).group(1).split(', '),
        'in_shape': list(map(int, in_shape_re.search(log_content).group(1).split(','))),
        'spatio_kernel_enc': int(spatio_kernel_enc_re.search(log_content).group(1)),
        'spatio_kernel_dec': int(spatio_kernel_dec_re.search(log_content).group(1)),
        'hid_S': int(hid_S_re.search(log_content).group(1)),
        'hid_T': int(hid_T_re.search(log_content).group(1)),
        'N_T': int(N_T_re.search(log_content).group(1)),
        'N_S': int(N_S_re.search(log_content).group(1))
    }

    for epoch_num, data in training_epochs.items():
        csv_line[f'epoch_{epoch_num}_steps'] = data['steps']
        csv_line[f'epoch_{epoch_num}_lr'] = data['lr']
        csv_line[f'epoch_{epoch_num}_train_loss'] = data['train_loss']
        csv_line[f'epoch_{epoch_num}_vali_loss'] = data['vali_loss']
        csv_line[f'epoch_{epoch_num}_train_seconds'] = data['train_seconds']
        csv_line[f'epoch_{epoch_num}_val_seconds'] = data['val_seconds']
        csv_line[f'epoch_{epoch_num}_mse'] = data['mse']
        csv_line[f'epoch_{epoch_num}_mae'] = data['mae']
        csv_line[f'epoch_{epoch_num}_ssim'] = data['ssim']

    return csv_line


def write_csv(parsed_data, output_csv):
    new_data_df = pd.DataFrame([parsed_data])

    if os.path.isfile(output_csv):
        existing_data_df = pd.read_csv(output_csv)

        combined_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
    else:
        new_data_df.to_csv(output_csv, index=False)

def main():
    parser = argparse.ArgumentParser(description="Parse log file and output to CSV.")
    parser.add_argument('log_file', help="The input log file to be parsed.")
    parser.add_argument('output_csv', help="The output CSV file to write the parsed data.")

    args = parser.parse_args()

    parsed_data = parse_log(args.log_file)
    write_csv(parsed_data, args.output_csv)


if __name__ == "__main__":
    main()
