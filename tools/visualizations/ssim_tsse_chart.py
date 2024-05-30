import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim, mean_squared_error
import cv2
import time

from openstl.api import BaseExperiment


def calculate_tsse(actual, forecast):
    squared_errors = (actual - forecast) ** 2
    return np.sum(squared_errors)


def calculate_mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    non_zero_actual = actual != 0
    return np.mean(np.abs((actual[non_zero_actual] - forecast[non_zero_actual]) / actual[non_zero_actual])) * 100


def save_plot(metrics, save_file):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Extracting plate_sizes from the metrics keys and sorting them for plotting
    plate_sizes = sorted(metrics.keys())
    ssim_values = [metrics[size]['SSIM'] for size in plate_sizes]
    tsse_values = [metrics[size]['TSSE'] for size in plate_sizes]

    color = 'tab:blue'
    ax1.set_xlabel('Plate Count')
    ax1.set_ylabel('SSIM', color=color)
    ax1.plot(plate_sizes, ssim_values, label='SSIM', marker='o', linestyle='-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('TSSE', color=color)  # we already handled the x-label with ax1
    ax2.plot(plate_sizes, tsse_values, label='TSSE', marker='o', linestyle='-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_yscale('log')
    ax2.set_yscale('log')

    # Optional: Add a title and a legend
    fig.tight_layout()  # to ensure the right y-label is not clipped
    fig.suptitle('SSIM and TSSE across plate counts', fontsize=16, va='top')
    fig.subplots_adjust(top=0.88)  # Adjust top to make space for suptitle

    # Saving the plot to the specified file
    plt.savefig(save_file)
    plt.close()


def calculate_mean_metrics(trues, preds):
    # Assuming trues and preds are of shape (n_rows, n_images_per_row, height, width)
    n_columns = trues.shape[1]
    metrics = {'SSIM': [], 'MSE': [], 'MAE': [], 'MAPE': [], 'TSSE': []}

    for column in range(n_columns):
        ssim_values, mse_values, mae_values, mape_values, tsse_values = [], [], [], [], []
        for row in range(trues.shape[0]):
            true_img = trues[row, column].squeeze()
            pred_img = preds[row, column].squeeze()

            ssim_values.append(ssim(true_img, pred_img))
            mse_values.append(mean_squared_error(true_img, pred_img))
            mae_values.append(np.mean(np.abs(true_img - pred_img)))
            mape_values.append(calculate_mape(true_img, pred_img))
            tsse_values.append(calculate_tsse(true_img, pred_img))

        # Calculate the mean of each metric for the current column
        metrics['SSIM'].append(np.mean(ssim_values))
        metrics['MSE'].append(np.mean(mse_values))
        metrics['MAE'].append(np.mean(mae_values))
        metrics['MAPE'].append(np.mean(mape_values))
        metrics['TSSE'].append(np.mean(tsse_values))

    metrics['SSIM'] = np.mean(metrics['SSIM'])
    metrics['MSE'] = np.mean(metrics['MSE'])
    metrics['MAE'] = np.mean(metrics['MAE'])
    metrics['MAPE'] = np.mean(metrics['MAPE'])
    metrics['TSSE'] = np.mean(metrics['TSSE'])

    return metrics


def save_metrics_statistics(metrics, output_path):
    with open(output_path + '.txt', 'w') as f:
        for metric_name, column_values in metrics.items():
            for column_index, value in enumerate(column_values):
                # Writing metric statistics per column
                f.write(f"{metric_name} - Column {column_index + 1}: {value:.10f}\n")

    for metric_name, column_values in metrics.items():
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(column_values) + 1), column_values, marker='o', linestyle='-', color='b')
        plt.xlabel('Column Number')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Averages per Column')
        plt.grid(True)

        plot_output_path = output_path + f'_{metric_name}.png'
        plt.savefig(plot_output_path)
        plt.close()


def load_data(ex_name, pre_seq_length, aft_seq_length, work_path="./work_dirs/"):
    save_folder = f'{work_path}{ex_name}/saved'

    inputs = np.load(f'{save_folder}/inputs.npy')
    preds = np.load(f'{save_folder}/preds.npy')
    trues = np.load(f'{save_folder}/trues.npy')

    return calculate_mean_metrics(trues, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chart the SSIM for each plate value at for the ex.")
    parser.add_argument("save_path", type=str, help="Save path")
    parser.add_argument("work_path", type=str, default="./work_dirs/", help="Work path")
    parser.add_argument("ex_name", type=str, help="Ex name")
    parser.add_argument("plate_size", type=int, help="Plate size")
    parser.add_argument("plate_sizes", nargs='+', type=int, help="Plate sizes")
    args = parser.parse_args()

    metrics = {}

    for plate_size in args.plate_sizes:
        ex_name = args.ex_name.replace(f'{args.plate_size}plates', f'{plate_size}plates')
        print(f'Starting run for ex {ex_name}')
        start_time = time.time()

        metrics[plate_size] = load_data(ex_name, 10, 10, args.work_path)

        print(f'Finished run for ex {ex_name}, took {time.time() - start_time} seconds')

    save_plot(metrics, args.save_path)
