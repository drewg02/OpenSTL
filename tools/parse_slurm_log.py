import re
import pandas as pd
from datetime import timedelta
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

hostname_re = re.compile(r'Running on hostname:\s*(.*)')
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
training_length_re = re.compile(r'(train loss: \d+\.\d+ \| data time: \d+\.\d+: 100%\|██████████\| \d+/\d+ \[\d+:\d+<\d+:\d+, *\d+\.\d+(?:it/s|s/it)\]\n?)+')
validation_metrics_re = re.compile(
    r'\[>{7,}\] \d+/\d+, [\d.]+ task/s, elapsed: (\d+)s, ETA:.*?mse:([\d.]+), mae:([\d.]+), ssim:([\d.]+)',
    re.MULTILINE)
training_time_re = re.compile(r'Training time: (\d+) days, (\d+):(\d+):(\d+)')

def parse_log(log_files):
    status = "FINISHED"
    job_id = os.path.basename(log_files[0]).split('-')[-1].split('.')[0]

    log_content = ""
    for log_file in log_files:
        with open(log_file, 'r') as file:
            log_content += file.read()

    if "FAILED" in log_content:
        status = "FAILED"

    rank_matches = re.findall(rank_pattern_re, log_content)
    num_gpus = len(rank_matches)

    hostname_match = hostname_re.search(log_content)
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

    if not hostname_match or not config_file_match or not ex_name_match or not seed_match or not batch_size_match or not val_batch_size_match or not pre_seq_length_match or not aft_seq_length_match or not method_match or not config_file_match or not epoch_match or not in_shape_match:
        status = "FAILED"

    hostname = hostname_match.group(1)
    world_size = world_size_match.group(1) if world_size_match else 1

    training_epochs = {}
    training_metrics_matches = list(re.finditer(training_info_re, log_content))
    training_length_matches = list(re.finditer(training_length_re, log_content))
    validation_metrics_matches = list(re.finditer(validation_metrics_re, log_content))

    test_time, test_mse, test_mae, test_ssim = None, None, None, None
    if len(training_metrics_matches) < 1 or (len(training_metrics_matches) != len(training_length_matches) != len(validation_metrics_matches) - 1):
        status = "FAILED"
    else:
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

            train_length = train_length_match.group(0).split('\n')[-2].strip()
            match = re.search(r"\[(\d{2}):(\d{2})<\d{2}:\d{2},", train_length)

            total_seconds = None
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
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

        final_test_match = validation_metrics_matches[-1]
        if final_test_match and "val" not in final_test_match.group(0):
            test_time = int(final_test_match.group(1))
            test_mse = float(final_test_match.group(2))
            test_mae = float(final_test_match.group(3))
            test_ssim = float(final_test_match.group(4))

    training_time_match = training_time_re.search(log_content)
    training_time_seconds = None
    if training_time_match:
        training_time_seconds = int(timedelta(days=int(training_time_match.group(1)),
                                          hours=int(training_time_match.group(2)),
                                          minutes=int(training_time_match.group(3)),
                                          seconds=int(training_time_match.group(4))).total_seconds())
    elif status == "FINISHED":
        status = "UNFINISHED"

    device_match = device_re.search(log_content)
    dist_match = dist_re.search(log_content)
    display_step_match = display_step_re.search(log_content)
    res_dir_match = res_dir_re.search(log_content)
    ex_name_match = ex_name_re.search(log_content)
    use_gpu_match = use_gpu_re.search(log_content)
    fp16_match = fp16_re.search(log_content)
    torchscript_match = torchscript_re.search(log_content)
    seed_match = seed_re.search(log_content)
    diff_seed_match = diff_seed_re.search(log_content)
    fps_match = fps_re.search(log_content)
    empty_cache_match = empty_cache_re.search(log_content)
    find_unused_parameters_match = find_unused_parameters_re.search(log_content)
    broadcast_buffers_match = broadcast_buffers_re.search(log_content)
    resume_from_match = resume_from_re.search(log_content)
    auto_resume_match = auto_resume_re.search(log_content)
    test_match = test_re.search(log_content)
    inference_match = inference_re.search(log_content)
    deterministic_match = deterministic_re.search(log_content)
    launcher_match = launcher_re.search(log_content)
    local_rank_match = local_rank_re.search(log_content)
    port_match = port_re.search(log_content)
    batch_size_match = batch_size_re.search(log_content)
    val_batch_size_match = val_batch_size_re.search(log_content)
    num_workers_match = num_workers_re.search(log_content)
    data_root_match = data_root_re.search(log_content)
    dataname_match = dataname_re.search(log_content)
    pre_seq_length_match = pre_seq_length_re.search(log_content)
    aft_seq_length_match = aft_seq_length_re.search(log_content)
    total_length_match = total_length_re.search(log_content)
    use_augment_match = use_augment_re.search(log_content)
    use_prefetcher_match = use_prefetcher_re.search(log_content)
    drop_last_match = drop_last_re.search(log_content)
    method_match = method_re.search(log_content)
    config_file_match = config_file_re.search(log_content)
    model_type_match = model_type_re.search(log_content)
    drop_match = drop_re.search(log_content)
    drop_path_match = drop_path_re.search(log_content)
    overwrite_match = overwrite_re.search(log_content)
    epoch_match = epoch_re.search(log_content)
    checkpoint_interval_match = checkpoint_interval_re.search(log_content)
    log_step_match = log_step_re.search(log_content)
    opt_match = opt_re.search(log_content)
    opt_eps_match = opt_eps_re.search(log_content)
    opt_betas_match = opt_betas_re.search(log_content)
    momentum_match = momentum_re.search(log_content)
    weight_decay_match = weight_decay_re.search(log_content)
    clip_grad_match = clip_grad_re.search(log_content)
    clip_mode_match = clip_mode_re.search(log_content)
    early_stop_epoch_match = early_stop_epoch_re.search(log_content)
    no_display_method_info_match = no_display_method_info_re.search(log_content)
    sched_match = sched_re.search(log_content)
    lr_match = lr_re.search(log_content)
    lr_k_decay_match = lr_k_decay_re.search(log_content)
    warmup_lr_match = warmup_lr_re.search(log_content)
    min_lr_match = min_lr_re.search(log_content)
    final_div_factor_match = final_div_factor_re.search(log_content)
    warmup_epoch_match = warmup_epoch_re.search(log_content)
    decay_epoch_match = decay_epoch_re.search(log_content)
    decay_rate_match = decay_rate_re.search(log_content)
    filter_bias_and_bn_match = filter_bias_and_bn_re.search(log_content)
    datafile_in_match = datafile_in_re.search(log_content)
    saved_path_match = saved_path_re.search(log_content)
    metrics_match = metrics_re.search(log_content)
    in_shape_match = in_shape_re.search(log_content)
    spatio_kernel_enc_match = spatio_kernel_enc_re.search(log_content)
    spatio_kernel_dec_match = spatio_kernel_dec_re.search(log_content)
    hid_S_match = hid_S_re.search(log_content)
    hid_T_match = hid_T_re.search(log_content)
    N_T_match = N_T_re.search(log_content)
    N_S_match = N_S_re.search(log_content)

    csv_line = {
        'jobid': job_id,
        'status': status,
        'hostname': hostname,
        'world_size': world_size,
        'training_time_seconds': training_time_seconds,
        'test_time': test_time,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_ssim': test_ssim,
        'num_gpus': num_gpus,
        'device': device_match.group(1) if device_match else None,
        'dist': dist_match.group(1) == 'True' if dist_match else None,
        'display_step': int(display_step_match.group(1)) if display_step_match else None,
        'res_dir': res_dir_match.group(1) if res_dir_match else None,
        'ex_name': ex_name_match.group(1) if ex_name_match else None,
        'use_gpu': use_gpu_match.group(1) == 'True' if use_gpu_match else None,
        'fp16': fp16_match.group(1) == 'False' if fp16_match else None,
        'torchscript': torchscript_match.group(1) == 'False' if torchscript_match else None,
        'seed': int(seed_match.group(1)) if seed_match else None,
        'diff_seed': diff_seed_match.group(1) == 'False' if diff_seed_match else None,
        'fps': fps_match.group(1) == 'False' if fps_match else None,
        'empty_cache': empty_cache_match.group(1) == 'True' if empty_cache_match else None,
        'find_unused_parameters': find_unused_parameters_match.group(
            1) == 'False' if find_unused_parameters_match else None,
        'broadcast_buffers': broadcast_buffers_match.group(1) == 'True' if broadcast_buffers_match else None,
        'resume_from': resume_from_match.group(1) if resume_from_match else None,
        'auto_resume': auto_resume_match.group(1) == 'False' if auto_resume_match else None,
        'test': test_match.group(1) == 'False' if test_match else None,
        'inference': inference_match.group(1) == 'False' if inference_match else None,
        'deterministic': deterministic_match.group(1) == 'False' if deterministic_match else None,
        'launcher': launcher_match.group(1) if launcher_match else None,
        'local_rank': int(local_rank_match.group(1)) if local_rank_match else None,
        'port': int(port_match.group(1)) if port_match else None,
        'batch_size': int(batch_size_match.group(1)) if batch_size_match else None,
        'val_batch_size': int(val_batch_size_match.group(1)) if val_batch_size_match else None,
        'num_workers': int(num_workers_match.group(1)) if num_workers_match else None,
        'data_root': data_root_match.group(1) if data_root_match else None,
        'dataname': dataname_match.group(1) if dataname_match else None,
        'pre_seq_length': int(pre_seq_length_match.group(1)) if pre_seq_length_match else None,
        'aft_seq_length': int(aft_seq_length_match.group(1)) if aft_seq_length_match else None,
        'total_length': int(total_length_match.group(1)) if total_length_match else None,
        'use_augment': use_augment_match.group(1) == 'False' if use_augment_match else None,
        'use_prefetcher': use_prefetcher_match.group(1) == 'False' if use_prefetcher_match else None,
        'drop_last': drop_last_match.group(1) == 'False' if drop_last_match else None,
        'method': method_match.group(1) if method_match else None,
        'config_file': config_file_match.group(1) if config_file_match else None,
        'model_type': model_type_match.group(1) if model_type_match else None,
        'drop': float(drop_match.group(1)) if drop_match else None,
        'drop_path': int(drop_path_match.group(1)) if drop_path_match else None,
        'overwrite': overwrite_match.group(1) == 'False' if overwrite_match else None,
        'epoch': int(epoch_match.group(1)) if epoch_match else None,
        'checkpoint_interval': checkpoint_interval_match.group(1) if checkpoint_interval_match else None,
        'log_step': int(log_step_match.group(1)) if log_step_match else None,
        'opt': opt_match.group(1) if opt_match else None,
        'opt_eps': opt_eps_match.group(1) if opt_eps_match else None,
        'opt_betas': opt_betas_match.group(1) if opt_betas_match else None,
        'momentum': float(momentum_match.group(1)) if momentum_match else None,
        'weight_decay': float(weight_decay_match.group(1)) if weight_decay_match else None,
        'clip_grad': clip_grad_match.group(1) if clip_grad_match else None,
        'clip_mode': clip_mode_match.group(1) if clip_mode_match else None,
        'early_stop_epoch': early_stop_epoch_match.group(1) if early_stop_epoch_match else None,
        'no_display_method_info': no_display_method_info_match.group(
            1) == 'False' if no_display_method_info_match else None,
        'sched': sched_match.group(1) if sched_match else None,
        'lr': float(lr_match.group(1)) if lr_match else None,
        'lr_k_decay': float(lr_k_decay_match.group(1)) if lr_k_decay_match else None,
        'warmup_lr': float(warmup_lr_match.group(1)) if warmup_lr_match else None,
        'min_lr': float(min_lr_match.group(1)) if min_lr_match else None,
        'final_div_factor': float(final_div_factor_match.group(1)) if final_div_factor_match else None,
        'warmup_epoch': int(warmup_epoch_match.group(1)) if warmup_epoch_match else None,
        'decay_epoch': int(decay_epoch_match.group(1)) if decay_epoch_match else None,
        'decay_rate': float(decay_rate_match.group(1)) if decay_rate_match else None,
        'filter_bias_and_bn': filter_bias_and_bn_match.group(1) == 'False' if filter_bias_and_bn_match else None,
        'datafile_in': datafile_in_match.group(1) if datafile_in_match else None,
        'saved_path': saved_path_match.group(1) if saved_path_match else None,
        'metrics': metrics_match.group(1).split(', ') if metrics_match else None,
        'in_shape': list(map(int, in_shape_match.group(1).split(','))) if in_shape_match else None,
        'spatio_kernel_enc': int(spatio_kernel_enc_match.group(1)) if spatio_kernel_enc_match else None,
        'spatio_kernel_dec': int(spatio_kernel_dec_match.group(1)) if spatio_kernel_dec_match else None,
        'hid_S': int(hid_S_match.group(1)) if hid_S_match else None,
        'hid_T': int(hid_T_match.group(1)) if hid_T_match else None,
        'N_T': int(N_T_match.group(1)) if N_T_match else None,
        'N_S': int(N_S_match.group(1)) if N_S_match else None
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
        existing_data_df = pd.read_csv(output_csv, low_memory=False)

        new_data_df = new_data_df.dropna(axis=1, how='all')
        existing_data_df = existing_data_df.dropna(axis=1, how='all')

        combined_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
    else:
        new_data_df.to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Parse log files and output to CSV.")
    parser.add_argument('log_dir', help="The directory containing log files to be parsed.")
    parser.add_argument('output_csv', help="The output CSV file to write the parsed data.")

    args = parser.parse_args()

    log_files = defaultdict(list)

    for file_name in os.listdir(args.log_dir):
        if file_name.endswith('.out') or file_name.endswith('.err'):
            job_id = file_name.split('-')[-1].split('.')[0]
            log_files[job_id].append(os.path.join(args.log_dir, file_name))

    for job_id, files in tqdm(log_files.items(), desc="Processing log files"):
        parsed_data = parse_log(files)
        write_csv(parsed_data, args.output_csv)


if __name__ == "__main__":
    main()
