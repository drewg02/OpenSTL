from openstl.utils import show_video_line, show_video_gif_multiple
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage.metrics import structural_similarity as ssim, mean_squared_error
import imageio

def load_results(ex_name, res_dir='work_dirs'):
    save_folder = f'./{res_dir}/{ex_name}/saved'
    inputs = np.load(f'{save_folder}/inputs.npy')
    trues = np.load(f'{save_folder}/trues.npy')
    preds = np.load(f'{save_folder}/preds.npy')
    return inputs, trues, preds


def save_result_images(inputs, preds, trues, simulation, save_folder):
    print(inputs.shape, preds.shape, trues.shape)
    for idx, (input_sample, pred_sample, true_sample) in enumerate(zip(inputs, preds, trues)):
        sample_folder = os.path.join(save_folder, f'sample{idx}')
        os.makedirs(sample_folder, exist_ok=True)

        # Save each input frame
        for frame_idx, frame in enumerate(input_sample):
            input_path = os.path.join(sample_folder, f'input{frame_idx + 1}.png')
            show_video_line(frame[np.newaxis, :], ncols=1, vmax=simulation.vmax, cmap=simulation.cmap, out_path=input_path)

        # Save each prediction frame
        for frame_idx, frame in enumerate(pred_sample):
            pred_path = os.path.join(sample_folder, f'predicted{frame_idx + 1}.png')
            show_video_line(frame[np.newaxis, :], ncols=1, vmax=simulation.vmax, cmap=simulation.cmap, out_path=pred_path)

        # Save each ground truth frame
        for frame_idx, frame in enumerate(true_sample):
            true_path = os.path.join(sample_folder, f'ground_truth{frame_idx + 1}.png')
            show_video_line(frame[np.newaxis, :], ncols=1, vmax=simulation.vmax, cmap=simulation.cmap, out_path=true_path)


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
                            format='png', out_path=None):
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


def show_video_line_ssim(inputs, trues, preds, diff, ncols, vmax=1.0, vmin=0, cmap='gray', diff_cmap='gray', norm=None, cbar=False,
                            format='png', out_path=None):
    nrows = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.25 * ncols, 6.5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if len(inputs.shape) > 3:
        inputs = inputs.swapaxes(1, 2).swapaxes(2, 3)
    if len(trues.shape) > 3:
        trues = trues.swapaxes(1, 2).swapaxes(2, 3)
    if len(preds.shape) > 3:
        preds = preds.swapaxes(1, 2).swapaxes(2, 3)
    if len(diff.shape) > 3:
        diff = diff.swapaxes(1, 2).swapaxes(2, 3)

    images = []
    for t in range(ncols):
        ax_inputs = axes[0, t]
        inputs_img = trues[t]
        im_inputs = ax_inputs.imshow(inputs_img, cmap=cmap, norm=norm)
        ax_inputs.axis('off')
        im_inputs.set_clim(vmin, vmax)

        ax_true = axes[1, t]
        true_img = trues[t]
        im_true = ax_true.imshow(true_img, cmap=cmap, norm=norm)
        ax_true.axis('off')
        im_true.set_clim(vmin, vmax)

        ax_pred = axes[2, t]
        pred_img = preds[t]
        im_pred = ax_pred.imshow(pred_img, cmap=cmap, norm=norm)
        ax_pred.axis('off')
        im_pred.set_clim(vmin, vmax)

        ax_diff = axes[3, t]
        diff_img = preds[t]
        im_diff = ax_diff.imshow(diff_img, cmap=diff_cmap, norm=norm)
        ax_diff.axis('off')
        im_diff.set_clim(vmin, vmax)

        images.append(im_true)
        images.append(im_pred)

        true = trues[t].squeeze()
        pred = preds[t].squeeze()

        ssim_value = ssim(true, pred)

        if t == 0:
            ax_inputs.text(x=0, y=0.5, s='Inputs', rotation=90, size=20, va='center', ha='right', transform=ax_inputs.transAxes)
            ax_true.text(x=0, y=0.5, s='Ground truth', rotation=90, size=20, va='center', ha='right', transform=ax_true.transAxes)
            ax_pred.text(x=0, y=0.5, s='Predicted', rotation=90, size=20, va='center', ha='right', transform=ax_pred.transAxes)
            ax_diff.text(x=0, y=0.5, s='Difference', rotation=90, size=24, va='center', ha='right', transform=ax_diff.transAxes)

        ax_pred.text(0.5, -0.2, f"SSIM: {ssim_value:.5f}", size=24, ha="center",
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

def save_result_visualizations(res_dir, ex_name, simulation, normalized=True, result_suffix=""):
    save_folder = f'./{res_dir}/{ex_name}/saved'

    inputs = np.load(f'{save_folder}/inputs{result_suffix}.npy')
    trues = np.load(f'{save_folder}/trues{result_suffix}.npy')
    preds = np.load(f'{save_folder}/preds{result_suffix}.npy')

    vmax = (1 if normalized else simulation.vmax)
    prefix = simulation.__name__.lower()

    pre_seq_length, aft_seq_length = inputs.shape[1], trues.shape[1]

    example_idx = 0

    show_video_line(inputs[example_idx], ncols=pre_seq_length, vmax=vmax, vmin=0, cbar=False, format='png',
                    cmap=simulation.cmap,
                    out_path=f'{save_folder}/{prefix}_input{result_suffix}.png')
    show_video_line(preds[example_idx], ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False, format='png',
                    cmap=simulation.cmap,
                    out_path=f'{save_folder}/{prefix}_pred{result_suffix}.png')
    show_video_line(trues[example_idx], ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False, format='png',
                    cmap=simulation.cmap,
                    out_path=f'{save_folder}/{prefix}_true{result_suffix}.png')

    diff = np.abs(preds[example_idx] - trues[example_idx])
    show_video_line(diff, ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False, format='png', cmap=simulation.diff_cmap,
                    out_path=f'{save_folder}/{prefix}_diff{result_suffix}.png')

    show_video_gif_multiple(inputs[example_idx], trues[example_idx], preds[example_idx], vmax=vmax, vmin=0,
                            cmap=simulation.cmap,
                            out_path=f'{save_folder}/{prefix}{result_suffix}.gif')

    metrics = calculate_column_metrics(trues, preds)
    output_path = f'{save_folder}/metrics_statistics'
    save_metrics_statistics(metrics, output_path)

    for idx in range(0, min(trues.shape[0], 5)):
        show_video_line_metrics(metrics, trues[idx], preds[idx], ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False,
                                format='png',
                                cmap=simulation.cmap,
                                out_path=f'{save_folder}/{prefix}_metrics_{idx}.png')
        show_video_line_ssim(inputs[idx], trues[idx], preds[idx], preds[idx] - trues[idx], ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False,
                                format='png',
                                cmap=simulation.cmap,
                                diff_cmap=simulation.diff_cmap,
                                out_path=f'{save_folder}/{prefix}_ssim_{idx}.png')

    metric_files = ['mse.npy', 'mae.npy', 'lr.npy', 'train_loss.npy', 'vali_loss.npy']

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


def save_dataset_visualization(dataset, simulation_class, start_index=0, end_index=1, start_frame_index=0, end_frame_index=None, single=False, save_path="", normalized=True):
    vmax = 1 if normalized else simulation_class.vmax
    prefix = simulation_class.__name__.lower()

    if isinstance(dataset, dict):
        process_dataset(dataset, start_index, end_index, start_frame_index, end_frame_index, single, prefix, vmax, simulation_class.cmap, save_path)
    elif isinstance(dataset, np.ndarray):
        process_array(dataset, start_index, end_index, start_frame_index, end_frame_index, single, prefix, vmax, simulation_class.cmap, save_path)
    else:
        raise ValueError("Dataset must be either a dictionary of numpy arrays or a numpy array.")

def process_dataset(dataset, start_index, end_index, start_frame_index, end_frame_index, single, prefix, vmax, cmap, save_path):
    for key, value in dataset.items():
        process_data(value, key, start_index, end_index, start_frame_index, end_frame_index, single, prefix, vmax, cmap, save_path)

def process_array(data, start_index, end_index, start_frame_index, end_frame_index, single, prefix, vmax, cmap, save_path):
    process_data(data, None, start_index, end_index, start_frame_index, end_frame_index, single, prefix, vmax, cmap, save_path)

def process_data(data, key, start_index, end_index, start_frame_index, end_frame_index, single, prefix, vmax, cmap, save_path):
    length = data.shape[0]
    for index in range(min(start_index, length), min(end_index, length)):
        data_to_save = data[index][start_frame_index:end_frame_index]
        save_data(data_to_save, key, index, start_frame_index, end_frame_index, single, prefix, vmax, cmap, save_path)

def save_data(data, key, index, start_frame_index, end_frame_index, single, prefix, vmax, cmap, save_path):
    if single:
        for frame_index in range(data.shape[0]):
            frame_data = data[frame_index]
            save_frames(frame_data, key, prefix, f'sample-{index}', f'frame-{start_frame_index + frame_index}', vmax, cmap, save_path)
    else:
        save_frames(data, key, prefix, f'sample-{index}', f'frames-{start_frame_index}-to-{end_frame_index if end_frame_index else data.shape[0] - 1}', vmax, cmap, save_path)
        filename = f"{prefix}{f'_{key}' if key else ''}_{index}_frames-{start_frame_index}-to-{end_frame_index if end_frame_index else data.shape[0] - 1}"
        show_video_gif_single(data, vmax=vmax, vmin=0, cmap=cmap, out_path=f'{save_path}/{filename}')

def save_frames(data, key, prefix, index, frame_index, vmax, cmap, save_path):
    filename = f"{prefix}{f'_{key}' if key else ''}_{index}_{frame_index}"
    out_path = f'{save_path}/{filename}.png'
    show_video_line(data, ncols=data.shape[0], vmax=vmax, vmin=0, cbar=False, format='png', cmap=cmap, out_path=out_path)



def show_video_gif_single(data, vmax=0.6, vmin=0.0, cmap='gray', norm=None, out_path=None, use_rgb=False):
    """generate gif with a video sequence"""
    def swap_axes(x):
        if len(x.shape) > 3:
            return x.swapaxes(1,2).swapaxes(2,3)
        else: return x

    data = swap_axes(data)
    images = []
    for i in range(data.shape[0]):
        fig, ax = plt.subplots()
        if use_rgb:
            img = ax.imshow(cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB))
        else:
            img = ax.imshow(data[i], cmap=cmap, norm=norm)
        img.set_clim(vmin, vmax)
        ax.axis('off')
        plt.savefig('./tmp.png', bbox_inches='tight', format='png')
        images.append(imageio.imread('./tmp.png'))
        plt.close()
    os.remove('./tmp.png')

    if out_path is not None:
        if not out_path.endswith('gif'):
            out_path += '.gif'
        imageio.mimsave(out_path, images)