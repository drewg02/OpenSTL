from openstl.utils import show_video_line, show_video_gif_multiple
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from skimage.metrics import structural_similarity as ssim, mean_squared_error


def show_video_line_tsse(trues, preds, ncols, vmax=0.6, vmin=0.0, cmap='gray', norm=None, cbar=False, format='png', out_path=None, use_rgb=False):
    """generate images with a video sequence and display TSSE between trues and preds"""
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
        true_img = cv2.cvtColor(trues[t], cv2.COLOR_BGR2RGB) if use_rgb else trues[t]
        im_true = ax_true.imshow(true_img, cmap=cmap, norm=norm)
        ax_true.axis('off')
        im_true.set_clim(vmin, vmax)

        ax_pred = axes[1, t]
        pred_img = cv2.cvtColor(preds[t], cv2.COLOR_BGR2RGB) if use_rgb else preds[t]
        im_pred = ax_pred.imshow(pred_img, cmap=cmap, norm=norm)
        ax_pred.axis('off')
        im_pred.set_clim(vmin, vmax)

        tsse = calculate_tsse(trues[t], preds[t])
        ax_pred.text(0.5, -0.1, f"TSSE: {tsse:.2f}", size=12, ha="center", transform=ax_pred.transAxes)

        images.append(im_true)
        images.append(im_pred)

    if cbar and ncols > 1:
        cbaxes = fig.add_axes([0.9, 0.15, 0.04 / ncols, 0.7 * nrows])
        cbar = fig.colorbar(im_pred, ax=axes.ravel().tolist(), shrink=0.1, cax=cbaxes)

    # plt.show()
    if out_path is not None:
        fig.savefig(out_path, format=format, pad_inches=0, bbox_inches='tight')
    plt.close()


def calculate_tsse(actual, forecast):
    squared_errors = (actual - forecast) ** 2
    return np.sum(squared_errors)


def calculate_mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    non_zero_actual = actual != 0
    return np.mean(np.abs((actual[non_zero_actual] - forecast[non_zero_actual]) / actual[non_zero_actual])) * 100


def show_video_line_metrics(metrics, trues, preds, ncols, vmax=1.0, vmin=0, cmap='gray', norm=None, cbar=False,
                            format='png', out_path=None, use_rgb=False):
    nrows = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.25 * ncols, 6.5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if len(trues.shape) > 3:
        trues = trues.swapaxes(1, 2).swapaxes(2, 3)
    if len(preds.shape) > 3:
        preds = preds.swapaxes(1, 2).swapaxes(2, 3)

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
            ax_pred.text(0.5, text_y_start + i * text_y_offset, metric_text, size=12, ha="center",
                         transform=ax_pred.transAxes)

    if cbar and ncols > 1:
        cbaxes = fig.add_axes([0.9, 0.15, 0.04 / ncols, 0.7 * nrows])
        cbar = fig.colorbar(im_pred, ax=axes.ravel().tolist(), shrink=0.1, cax=cbaxes)

    if out_path is not None:
        fig.savefig(out_path, format=format, pad_inches=0, bbox_inches='tight')
    plt.close()

def plot_metric(metric, metric_file, folder_path):
    plt.figure()
    plt.plot(metric)
    plt.title(f"{metric_file.split('.')[0]} over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel(metric_file.split('.')[0])
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, f"{metric_file.replace('.npy', '.png')}"))


def plot_combined_loss(train_loss, vali_loss, folder_path):
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(vali_loss, label='Validation Loss')
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, 'loss.png'))


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

def save_result_visualizations(ex_name, pre_seq_length, aft_seq_length, simulation, normalized=True):
    save_folder = f'./work_dirs/{ex_name}/saved'

    inputs = np.load(f'{save_folder}/inputs.npy')
    preds = np.load(f'{save_folder}/preds.npy')
    trues = np.load(f'{save_folder}/trues.npy')

    vmax = (1 if normalized else simulation.vmax)
    prefix = simulation.__name__.lower()

    example_idx = 0

    show_video_line(inputs[example_idx], ncols=pre_seq_length, vmax=vmax, vmin=0, cbar=False, format='png',
                    cmap=simulation.cmap,
                    out_path=f'{save_folder}/{prefix}_input.png')
    show_video_line(preds[example_idx], ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False, format='png',
                    cmap=simulation.cmap,
                    out_path=f'{save_folder}/{prefix}_pred.png')
    show_video_line(trues[example_idx], ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False, format='png',
                    cmap=simulation.cmap,
                    out_path=f'{save_folder}/{prefix}_true.png')

    diff = np.abs(preds[example_idx] - trues[example_idx])
    show_video_line(diff, ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False, format='png', cmap=simulation.diff_cmap,
                    out_path=f'{save_folder}/{prefix}_diff.png')

    show_video_gif_multiple(inputs[example_idx], trues[example_idx], preds[example_idx], vmax=vmax, vmin=0,
                            cmap=simulation.cmap,
                            out_path=f'{save_folder}/{prefix}.gif')

    metrics = calculate_column_metrics(trues, preds)
    output_path = f'./work_dirs/{ex_name}/saved/metrics_statistics'
    save_metrics_statistics(metrics, output_path)

    for idx in range(0, min(trues.shape[0], 5)):
        show_video_line_metrics(metrics, trues[idx], preds[idx], ncols=aft_seq_length, vmax=212, vmin=0, cbar=False,
                                format='png',
                                cmap='Reds',
                                out_path=f'./work_dirs/{ex_name}/saved/2dplate_metrics_{idx}.png')

    # Metric filenames
    metric_files = ['mse.npy', 'mae.npy', 'lr.npy', 'train_loss.npy', 'vali_loss.npy']

    # Load and plot each metric
    train_loss, vali_loss = None, None
    for metric_file in metric_files:
        metric_path = os.path.join(save_folder, metric_file)
        if os.path.exists(metric_path):
            metric = np.load(metric_path)
            plot_metric(metric, metric_file, save_folder)
            if metric_file == 'train_loss.npy':
                train_loss = metric
            if metric_file == 'vali_loss.npy':
                vali_loss = metric
        else:
            print(f"Metric file {metric_file} not found in {save_folder}")

    # Plot combined train and validation loss
    if train_loss is not None and vali_loss is not None:
        plot_combined_loss(train_loss, vali_loss, save_folder)

def save_dataset_visualizations(dataset_path, pre_seq_length, aft_seq_length, simulation, save_path="", normalized=True):
    vmax = (1 if normalized else simulation.vmax)
    prefix = simulation.__name__.lower()

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    example_idx = 0
    for X in ['X_train', 'X_val', 'X_test']:
        show_video_line(dataset[X][example_idx], ncols=pre_seq_length, vmax=vmax, vmin=0, cbar=False, format='png',
                        cmap=simulation.cmap,
                        out_path=f'{save_path}/{prefix}_{X}_input{0}.png')
    for y in ['Y_train', 'Y_val', 'Y_test']:
        show_video_line(dataset[y][example_idx], ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False, format='png',
                        cmap=simulation.cmap,
                        out_path=f'{save_path}/{prefix}_{y}_true{0}.png')