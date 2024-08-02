import re
import pandas as pd
from datetime import timedelta
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

hostname_re = re.compile(r'^Running on hostname:.*\n(.*)', re.MULTILINE)
world_size_re = re.compile(r'^Distributed world_size=(\d+)', re.MULTILINE)
rank_pattern_re = re.compile(r'^Use distributed mode with GPUs: local rank=\d+', re.MULTILINE)
config_re = re.compile(r'^loading config from (.*) \.\.\.', re.MULTILINE)
device_re = re.compile(r'^device:\s*(.*)', re.MULTILINE)
dist_re = re.compile(r'^dist:\s*(True|False)', re.MULTILINE)
display_step_re = re.compile(r'^display_step:\s*(\d+)', re.MULTILINE)
res_dir_re = re.compile(r'^res_dir:\s*(.*)', re.MULTILINE)
ex_name_re = re.compile(r'^ex_name:\s*(.*)', re.MULTILINE)
use_gpu_re = re.compile(r'^use_gpu:\s*(True|False)', re.MULTILINE)
fp16_re = re.compile(r'^fp16:\s*(True|False)', re.MULTILINE)
torchscript_re = re.compile(r'^torchscript:\s*(True|False)', re.MULTILINE)
seed_re = re.compile(r'^seed:\s*(\d+)', re.MULTILINE)
diff_seed_re = re.compile(r'^diff_seed:\s*(True|False)', re.MULTILINE)
fps_re = re.compile(r'^fps:\s*(True|False)', re.MULTILINE)
empty_cache_re = re.compile(r'^empty_cache:\s*(True|False)', re.MULTILINE)
find_unused_parameters_re = re.compile(r'^find_unused_parameters:\s*(True|False)', re.MULTILINE)
broadcast_buffers_re = re.compile(r'^broadcast_buffers:\s*(True|False)', re.MULTILINE)
resume_from_re = re.compile(r'^resume_from:\s*(.*)', re.MULTILINE)
auto_resume_re = re.compile(r'^auto_resume:\s*(True|False)', re.MULTILINE)
test_re = re.compile(r'^test:\s*(True|False)', re.MULTILINE)
inference_re = re.compile(r'^inference:\s*(True|False)', re.MULTILINE)
deterministic_re = re.compile(r'^deterministic:\s*(True|False)', re.MULTILINE)
launcher_re = re.compile(r'^launcher:\s*(.*)', re.MULTILINE)
local_rank_re = re.compile(r'^local_rank:\s*(\d+)', re.MULTILINE)
port_re = re.compile(r'^port:\s*(\d+)', re.MULTILINE)
batch_size_re = re.compile(r'^batch_size:\s*(\d+)', re.MULTILINE)
val_batch_size_re = re.compile(r'^val_batch_size:\s*(\d+)', re.MULTILINE)
num_workers_re = re.compile(r'^num_workers:\s*(\d+)', re.MULTILINE)
data_root_re = re.compile(r'^data_root:\s*(.*)', re.MULTILINE)
dataname_re = re.compile(r'^dataname:\s*(.*)', re.MULTILINE)
pre_seq_length_re = re.compile(r'^pre_seq_length:\s*(\d+)', re.MULTILINE)
aft_seq_length_re = re.compile(r'^aft_seq_length:\s*(\d+)', re.MULTILINE)
total_length_re = re.compile(r'^total_length:\s*(\d+)', re.MULTILINE)
use_augment_re = re.compile(r'^use_augment:\s*(True|False)', re.MULTILINE)
use_prefetcher_re = re.compile(r'^use_prefetcher:\s*(True|False)', re.MULTILINE)
drop_last_re = re.compile(r'^drop_last:\s*(True|False)', re.MULTILINE)
method_re = re.compile(r'^method:\s*(.*)', re.MULTILINE)
config_file_re = re.compile(r'^config_file:\s*(.*)', re.MULTILINE)
model_type_re = re.compile(r'^model_type:\s*(.*)', re.MULTILINE)
drop_re = re.compile(r'^drop:\s*(\d+\.\d+)', re.MULTILINE)
drop_path_re = re.compile(r'^drop_path:\s*(\d+)', re.MULTILINE)
overwrite_re = re.compile(r'^overwrite:\s*(True|False)', re.MULTILINE)
epoch_re = re.compile(r'^epoch:\s*(\d+)', re.MULTILINE)
checkpoint_interval_re = re.compile(r'^checkpoint_interval:\s*(.*)', re.MULTILINE)
log_step_re = re.compile(r'^log_step:\s*(\d+)', re.MULTILINE)
opt_re = re.compile(r'^opt:\s*(.*)', re.MULTILINE)
opt_eps_re = re.compile(r'^opt_eps:\s*(.*)', re.MULTILINE)
opt_betas_re = re.compile(r'^opt_betas:\s*(.*)', re.MULTILINE)
momentum_re = re.compile(r'^momentum:\s*(\d+\.\d+)', re.MULTILINE)
weight_decay_re = re.compile(r'^weight_decay:\s*(\d+\.\d+)', re.MULTILINE)
clip_grad_re = re.compile(r'^clip_grad:\s*(.*)', re.MULTILINE)
clip_mode_re = re.compile(r'^clip_mode:\s*(.*)', re.MULTILINE)
early_stop_epoch_re = re.compile(r'^early_stop_epoch:\s*(.*)', re.MULTILINE)
no_display_method_info_re = re.compile(r'^no_display_method_info:\s*(True|False)', re.MULTILINE)
sched_re = re.compile(r'^sched:\s*(.*)', re.MULTILINE)
lr_re = re.compile(r'^lr:\s*(\d+\.\d+)', re.MULTILINE)
lr_k_decay_re = re.compile(r'^lr_k_decay:\s*(\d+\.\d+)', re.MULTILINE)
warmup_lr_re = re.compile(r'^warmup_lr:\s*(\d+e-\d+)', re.MULTILINE)
min_lr_re = re.compile(r'^min_lr:\s*(\d+e-\d+)', re.MULTILINE)
final_div_factor_re = re.compile(r'^final_div_factor:\s*(\d+\.\d+)', re.MULTILINE)
warmup_epoch_re = re.compile(r'^warmup_epoch:\s*(\d+)', re.MULTILINE)
decay_epoch_re = re.compile(r'^decay_epoch:\s*(\d+)', re.MULTILINE)
decay_rate_re = re.compile(r'^decay_rate:\s*(\d+\.\d+)', re.MULTILINE)
filter_bias_and_bn_re = re.compile(r'^filter_bias_and_bn:\s*(True|False)', re.MULTILINE)
datafile_in_re = re.compile(r'^datafile_in:\s*(.*)', re.MULTILINE)
saved_path_re = re.compile(r'^saved_path:\s*(.*)', re.MULTILINE)
metrics_re = re.compile(r'^metrics:\s*\[(.*)\]', re.MULTILINE)
in_shape_re = re.compile(r'^in_shape:\s*\[(.*)\]', re.MULTILINE)
spatio_kernel_enc_re = re.compile(r'^spatio_kernel_enc:\s*(\d+)', re.MULTILINE)
spatio_kernel_dec_re = re.compile(r'^spatio_kernel_dec:\s*(\d+)', re.MULTILINE)
hid_S_re = re.compile(r'^hid_S:\s*(\d+)', re.MULTILINE)
hid_T_re = re.compile(r'^hid_T:\s*(\d+)', re.MULTILINE)
N_T_re = re.compile(r'^N_T:\s*(\d+)', re.MULTILINE)
N_S_re = re.compile(r'^N_S:\s*(\d+)', re.MULTILINE)

gpu_re = re.compile(r'GPU ([\d,]+): (.*?)(\d+)GB(?:\n|$)', re.MULTILINE)

training_info_re = re.compile(
    r'Epoch: (\d+), Steps: (\d+) \| Lr: ([\d.]+) \| Train Loss: ([\d.]+) \| Vali Loss: ([\d.]+)')
training_length_re = re.compile(
    r'(train loss: \d+\.\d+ \| data time: \d+\.\d+: 100%\|██████████\| \d+/\d+ \[\d+:\d+<\d+:\d+, *\d+\.\d+(?:it/s|s/it)\]\n?)+')
elapsed_re = re.compile(
    r'\[>{7,}\] \d+/\d+, [\d.]+ task/s, elapsed: (\d+)s, ETA:.*?$',
    re.MULTILINE)
validation_metrics_re = re.compile(r'mse:([\d.]+), mae:([\d.]+), ssim:([\d.]+)')
training_time_re = re.compile(r'Training time: (\d+) days, (\d+):(\d+):(\d+)')


def write_csv(all_data, output_csv):
    combined_df = pd.DataFrame(all_data)
    if os.path.isfile(output_csv):
        existing_data_df = pd.read_csv(output_csv, low_memory=False)
        combined_df = pd.concat([existing_data_df, combined_df], ignore_index=True)

    combined_df.to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Parse log files and output to CSV.")
    parser.add_argument('log_dir', help="The directory containing log files to be parsed.")
    parser.add_argument('output_csv', help="The output CSV file to write the parsed data.")

    args = parser.parse_args()

    log_files = defaultdict(list)
    all_data = []

    for file_name in os.listdir(args.log_dir):
        if file_name.endswith('.out') or file_name.endswith('.err'):
            job_id = file_name.split('-')[-1].split('.')[0]
            log_files[job_id].append(os.path.join(args.log_dir, file_name))

    for job_id, files in tqdm(log_files.items(), desc="Processing log files"):
        parsed_data = parse_log(files)
        all_data.append(parsed_data)

    write_csv(all_data, args.output_csv)


def parse_log(log_files):
    status = "FINISHED"
    failure_reason = "None"
    job_id = os.path.basename(log_files[0]).split('-')[-1].split('.')[0]

    log_content = ""
    for log_file in log_files:
        with open(log_file, 'r') as file:
            log_content += file.read()
            log_content += "\n"

    if "FAILED" in log_content:
        status = "FAILED"

    if "CUDA out of memory" in log_content:
        failure_reason = "OUT_OF_MEMORY"

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
    elapsed_matches = list(re.finditer(elapsed_re, log_content))

    test_time, test_mse, test_mae, test_ssim = None, None, None, None
    if len(training_metrics_matches) < 1 or (
            len(training_metrics_matches) != len(training_length_matches) != len(
        validation_metrics_matches) - 1 != epoch_match.group(1)):
        status = "UNFINISHED"
    else:
        for train_match, train_length_match, val_match, elapsed_match in zip(training_metrics_matches,
                                                                             training_length_matches,
                                                                             validation_metrics_matches[:-1],
                                                                             elapsed_matches[:-1]):
            epoch_num = int(train_match.group(1))
            steps = int(train_match.group(2))
            lr = float(train_match.group(3))
            train_loss = float(train_match.group(4))
            vali_loss = float(train_match.group(5))

            elapsed = elapsed_match.group(1)
            mse = float(val_match.group(1))
            mae = float(val_match.group(2))
            ssim = float(val_match.group(3))

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
        final_elapsed_match = elapsed_matches[-1]
        if final_test_match and "val" not in final_elapsed_match.group(0):
            test_time = int(final_elapsed_match.group(1))
            test_mse = float(final_test_match.group(1))
            test_mae = float(final_test_match.group(2))
            test_ssim = float(final_test_match.group(3))

    training_time_match = training_time_re.findall(log_content)
    total_gpu_training_time = 0
    training_time_seconds = 0

    if training_time_match and len(training_time_match) > 0:
        for training_time in training_time_match:
            total_seconds = int(timedelta(days=int(training_time[0]),
                                          hours=int(training_time[1]),
                                          minutes=int(training_time[2]),
                                          seconds=int(training_time[3])).total_seconds())
            total_gpu_training_time += total_seconds
            if total_seconds > training_time_seconds:
                training_time_seconds = total_seconds

    elif status == "FINISHED":
        status = "UNFINISHED"

    # GPU parsing
    gpu_matches = gpu_re.findall(log_content)
    gpus = {}
    for match in gpu_matches:
        gpu_ids = match[0].split(',')
        gpu_name = match[1]
        gpu_capacity = match[2]
        for gpu_id in gpu_ids:
            gpus[f'gpu_{gpu_id.strip()}'] = f'{gpu_name}{gpu_capacity}GB'
            gpus[f'gpu_{gpu_id.strip()}_capacity'] = gpu_capacity.strip()

    csv_line = {
        'jobid': job_id,
        'status': status,
        'failure_reason': failure_reason,
        'hostname': hostname,
        'world_size': world_size,
        'total_gpu_training_time': total_gpu_training_time,
        'training_time_seconds': training_time_seconds,
        'test_time': test_time,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_ssim': test_ssim,
        'num_gpus': num_gpus,
    }

    # Add GPU details to csv_line
    csv_line.update(gpus)

    # Add additional parsed data
    additional_data = [
        device_re, dist_re, display_step_re, res_dir_re, ex_name_re, use_gpu_re,
        fp16_re, torchscript_re, seed_re, diff_seed_re, fps_re, empty_cache_re,
        find_unused_parameters_re, broadcast_buffers_re, resume_from_re, auto_resume_re,
        test_re, inference_re, deterministic_re, launcher_re, local_rank_re, port_re,
        batch_size_re, val_batch_size_re, num_workers_re, data_root_re, dataname_re,
        pre_seq_length_re, aft_seq_length_re, total_length_re, use_augment_re,
        use_prefetcher_re, drop_last_re, method_re, config_file_re, model_type_re,
        drop_re, drop_path_re, overwrite_re, epoch_re, checkpoint_interval_re,
        log_step_re, opt_re, opt_eps_re, opt_betas_re, momentum_re, weight_decay_re,
        clip_grad_re, clip_mode_re, early_stop_epoch_re, no_display_method_info_re,
        sched_re, lr_re, lr_k_decay_re, warmup_lr_re, min_lr_re, final_div_factor_re,
        warmup_epoch_re, decay_epoch_re, decay_rate_re, filter_bias_and_bn_re,
        datafile_in_re, saved_path_re, metrics_re, in_shape_re, spatio_kernel_enc_re,
        spatio_kernel_dec_re, hid_S_re, hid_T_re, N_T_re, N_S_re
    ]

    for regex in additional_data:
        match = regex.search(log_content)
        if match:
            csv_line[regex.pattern.split(':')[0].replace("^", "")] = match.group(1)

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


if __name__ == "__main__":
    main()
