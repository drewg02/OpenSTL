import argparse
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim, mean_squared_error
import cv2
import time
from torch.nn.functional import mse_loss, l1_loss

from openstl.api import BaseExperiment
from openstl.utils import default_parser, show_video_line, show_video_gif_multiple, show_video_line_tsse

def calculate_tsse(actual, forecast):
    squared_errors = (actual - forecast) ** 2
    return np.sum(squared_errors)
    
def calculate_mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    non_zero_actual = actual != 0
    return np.mean(np.abs((actual[non_zero_actual] - forecast[non_zero_actual]) / actual[non_zero_actual])) * 100
    
def show_video_line_metrics(metrics, trues, preds, ncols, vmax=1.0, vmin=0, cmap='gray', norm=None, cbar=False, format='png', out_path=None, use_rgb=False):
    nrows = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.25 * ncols, 6.5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if len(trues.shape) > 3:
        trues = trues.swapaxes(1,2).swapaxes(2,3)
    if len(preds.shape) > 3:
        preds = preds.swapaxes(1,2).swapaxes(2,3)

    images = []
    for t in range(ncols):
        ax_true = axes[0, t]
        true_img = trues[t]
        im_true = ax_true.imshow(true_img, cmap=cmap, norm=norm)
        ax_true.axis('off')
        im_true.set_clim(vmin, vmax)

        ax_pred = axes[1, t]
        pred_img = preds[t]
        im_pred = ax_pred.imshow(pred_img, cmap=cmap, norm=norm)
        ax_pred.axis('off')
        im_pred.set_clim(vmin, vmax)
        
        images.append(im_true)
        images.append(im_pred)
        
        true = trues[t].squeeze()
        pred = preds[t].squeeze()
        
        ssim_value = ssim(true, pred)
        mse_value = mean_squared_error(true, pred)
        mae_value = np.mean(np.abs(true - pred))
        mape_value = calculate_mape(true, pred)
        tsse_value = calculate_tsse(true, pred)
        
        text_y_start = -0.1
        text_y_offset = -0.1 

        metrics_text = [
            f"SSIM: {ssim_value:.5f}",
            f"MSE: {mse_value:.5f}",
            f"MAE: {mae_value:.5f}",
            f"MAPE: {mape_value:.5f}",
            f"TSSE: {tsse_value:.5f}",
            f"MEAN SSIM: {metrics['SSIM'][t]:.5f}",
            f"MEAN MSE: {metrics['MSE'][t]:.5f}",
            f"MEAN MAE: {metrics['MAE'][t]:.5f}",
            f"MEAN MAPE: {metrics['MAPE'][t]:.5f}",
            f"MEAN TSSE: {metrics['TSSE'][t]:.5f}"
        ]
        
        for i, metric_text in enumerate(metrics_text):
            ax_pred.text(0.5, text_y_start + i * text_y_offset, metric_text, size=12, ha="center", transform=ax_pred.transAxes)

    if cbar and ncols > 1:
        cbaxes = fig.add_axes([0.9, 0.15, 0.04 / ncols, 0.7 * nrows])
        cbar = fig.colorbar(im_pred, ax=axes.ravel().tolist(), shrink=0.1, cax=cbaxes)

    if out_path is not None:
        fig.savefig(out_path, format=format, pad_inches=0, bbox_inches='tight')
    plt.close()
    
def calculate_column_metrics(trues, preds):
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
            
def save_visualizations(ex_name, pre_seq_length, aft_seq_length):
    save_folder = f'./work_dirs/{ex_name}/saved'

    inputs = np.load(f'{save_folder}/inputs.npy')
    preds = np.load(f'{save_folder}/preds.npy')
    trues = np.load(f'{save_folder}/trues.npy')
    
    metrics = calculate_column_metrics(trues, preds)
    output_path = f'./work_dirs/{ex_name}/saved/metrics_statistics'
    save_metrics_statistics(metrics, output_path)
    
    for idx in range(0, min(trues.shape[0], 5)):
        show_video_line_metrics(metrics, trues[idx], preds[idx], ncols=aft_seq_length, vmax=212, vmin=0, cbar=False, format='png',
                             cmap='Reds',
                             out_path=f'./work_dirs/{ex_name}/saved/2dplate_metrics_{idx}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the SSIM for each ex.")
    parser.add_argument("ex_names", nargs='+', type=str, help="Ex names")
    args = parser.parse_args()
    
    for ex_name in args.ex_names:
        print(f'Starting run for ex {ex_name}')
        start_time = time.time()
        
        save_visualizations(ex_name, 10, 10)
        
        print(f'Finished run for ex {ex_name}, took {time.time() - start_time} seconds')
