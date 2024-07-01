import re
import csv
from datetime import timedelta
import os
import argparse

# Define individual regular expressions to extract required information
hostname_re = re.compile(r'Running on hostname:\s*(.*)')
experiment_re = re.compile(r'---\[ Experiment: (.*) \]---')
world_size_re = re.compile(r'Distributed world_size=(\d+)')
config_re = re.compile(r'loading config from (.*)')
ex_name_re = re.compile(r'ex_name:\s*(.*?)\s*\t')
seed_re = re.compile(r'seed:\s*(\d+)\s*\t')
batch_size_re = re.compile(r'batch_size:\s*(\d+)\s*\t')
val_batch_size_re = re.compile(r'val_batch_size:\s*(\d+)\s*\t')
pre_seq_length_re = re.compile(r'pre_seq_length:\s*(\d+)\s*\t')
aft_seq_length_re = re.compile(r'aft_seq_length:\s*(\d+)\s*\t')
method_re = re.compile(r'method:\s*(\w+)\s*\t')
config_file_re = re.compile(r'config_file:\s*(.*?)\s*\t')
epoch_re = re.compile(r'epoch:\s*(\d+)\s*\t')
in_shape_re = re.compile(r'in_shape:\s*\[(.*?)\]')

training_info_re = re.compile(r'Epoch: (\d+), Steps: (\d+) \| Lr: ([\d.]+) \| Train Loss: ([\d.]+) \| Vali Loss: ([\d.]+)')
# Adjusted validation metrics regex pattern to handle different formats
validation_metrics_re = re.compile(r'\[>{7,}\] \d+/\d+, [\d.]+ task/s, elapsed: (\d+)s, ETA:.*?mse:([\d.]+), mae:([\d.]+), ssim:([\d.]+)', re.MULTILINE)
training_time_re = re.compile(r'Training time: (\d+) days, (\d+):(\d+):(\d+)')

def parse_log(log_file):
    # Extract job_id from filename
    job_id = os.path.basename(log_file).split('-')[1].split('.')[0]

    with open(log_file, 'r') as file:
        log_content = file.read()

    # Extract individual items
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

    # Log errors if any item is not found
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
        # Debugging: Print out the relevant section of the log content
        start_index = log_content.find("ex_name:")
        end_index = start_index + 1000  # Adjust the length as needed
        print("Relevant section of the log content:\n", log_content[start_index:end_index])

    if not hostname_match or not experiment_match or not config_file_match or not ex_name_match or not seed_match or not batch_size_match or not val_batch_size_match or not pre_seq_length_match or not aft_seq_length_match or not method_match or not config_file_match or not epoch_match or not in_shape_match:
        raise ValueError("Log file does not contain the necessary information.")

    # Extract values from matches
    hostname = hostname_match.group(1)
    experiment = experiment_match.group(1)
    world_size = world_size_match.group(1) if world_size_match else None
    ex_name = ex_name_match.group(1)
    seed = int(seed_match.group(1))
    batch_size = int(batch_size_match.group(1))
    val_batch_size = int(val_batch_size_match.group(1))
    pre_seq_length = int(pre_seq_length_match.group(1))
    aft_seq_length = int(aft_seq_length_match.group(1))
    method = method_match.group(1)
    config_file = config_file_match.group(1)
    epoch = int(epoch_match.group(1))
    in_shape = in_shape_match.group(1)

    # Extract training epochs information
    training_epochs = {}
    training_metrics_matches = list(re.finditer(training_info_re, log_content))
    validation_metrics_matches = list(re.finditer(validation_metrics_re, log_content))

    if len(training_metrics_matches) != len(validation_metrics_matches) - 1:
        print("Mismatch between the number of training and validation metric matches.")
        raise ValueError("Log file does not contain matching training and validation information.")

    for train_match, val_match in zip(training_metrics_matches, validation_metrics_matches[:-1]):
        epoch_num = int(train_match.group(1))
        steps = int(train_match.group(2))
        lr = float(train_match.group(3))
        train_loss = float(train_match.group(4))
        vali_loss = float(train_match.group(5))

        elapsed = int(val_match.group(1))
        mse = float(val_match.group(2))
        mae = float(val_match.group(3))
        ssim = float(val_match.group(4))

        training_epochs[epoch_num] = {
            'steps': steps,
            'lr': lr,
            'train_loss': train_loss,
            'vali_loss': vali_loss,
            'elapsed': elapsed,
            'mse': mse,
            'mae': mae,
            'ssim': ssim
        }

    # Extract training time
    training_time_match = training_time_re.search(log_content)
    training_time_seconds = int(timedelta(days=int(training_time_match.group(1)),
                                          hours=int(training_time_match.group(2)),
                                          minutes=int(training_time_match.group(3)),
                                          seconds=int(training_time_match.group(4))).total_seconds())

    # Extract final test results
    final_test_match = validation_metrics_matches[-1]
    test_time = int(final_test_match.group(1))
    test_mse = float(final_test_match.group(2))
    test_mae = float(final_test_match.group(3))
    test_ssim = float(final_test_match.group(4))

    # Summarize data for a single CSV line
    csv_line = {
        'jobid': job_id,
        'hostname': hostname,
        'experiment': experiment,
        'world_size': world_size,
        'ex_name': ex_name,
        'seed': seed,
        'batch_size': batch_size,
        'val_batch_size': val_batch_size,
        'pre_seq_length': pre_seq_length,
        'aft_seq_length': aft_seq_length,
        'method': method,
        'config_file': config_file,
        'epoch': epoch,
        'in_shape': in_shape,
        'training_time_seconds': training_time_seconds,
        'test_time': test_time,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_ssim': test_ssim,
    }

    for epoch_num, data in training_epochs.items():
        csv_line[f'epoch_{epoch_num}_steps'] = data['steps']
        csv_line[f'epoch_{epoch_num}_lr'] = data['lr']
        csv_line[f'epoch_{epoch_num}_train_loss'] = data['train_loss']
        csv_line[f'epoch_{epoch_num}_vali_loss'] = data['vali_loss']
        csv_line[f'epoch_{epoch_num}_elapsed'] = data['elapsed']
        csv_line[f'epoch_{epoch_num}_mse'] = data['mse']
        csv_line[f'epoch_{epoch_num}_mae'] = data['mae']
        csv_line[f'epoch_{epoch_num}_ssim'] = data['ssim']

    return csv_line

def write_csv(parsed_data, output_csv):
    file_exists = os.path.isfile(output_csv)
    fieldnames = list(parsed_data.keys())

    with open(output_csv, 'a' if file_exists else 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(parsed_data)

def main():
    parser = argparse.ArgumentParser(description="Parse log file and output to CSV.")
    parser.add_argument('log_file', help="The input log file to be parsed.")
    parser.add_argument('output_csv', help="The output CSV file to write the parsed data.")

    args = parser.parse_args()

    parsed_data = parse_log(args.log_file)
    write_csv(parsed_data, args.output_csv)

if __name__ == "__main__":
    main()
