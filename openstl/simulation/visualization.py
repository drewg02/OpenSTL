import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from openstl.simulation.utils import get_simulation_class


def load_data(datafolder, limit=100):
    folders = [f for f in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, f))]

    dataset = []
    for unique_id in folders:
        if len(dataset) >= limit:
            break

        files = [f for f in os.listdir(f'{datafolder}/{unique_id}') if f.endswith('.npy')]
        if len(files) < 1:
            continue

        files = [f for f in os.listdir(f'{datafolder}/{unique_id}') if f.endswith('.npy')]
        data = []
        for i in range(0, len(files)):
            data.append(np.load(f'{datafolder}/{unique_id}/{i}.npy'))

        data = np.array(data)
        data = np.squeeze(data)
        dataset.append(data)

    return np.array(dataset)

# Base function for plotting arrays
def plot_arrays(arrays, filename, rows=None, cols=None, wspace=10, hspace=10, dpi=100, formats=None, show=False,
                texts=None, text_positions=None, font_size=12, line_height=1.2, cmaps=None):
    if formats is None:
        formats = ['png']

    if len(arrays.shape) == 2:
        arrays = np.expand_dims(arrays, axis=0)

    num_arrays = arrays.shape[0]
    N = arrays.shape[1]

    if rows is None and cols is None:
        rows = 1
        cols = num_arrays
    elif rows is None and cols is not None:
        rows = (num_arrays + cols - 1) // cols
    elif cols is None and rows is not None:
        cols = (num_arrays + rows - 1) // rows

    N = N * (4 if N < 256 else 1)
    wspace = wspace * (4 if N < 256 else 1)
    hspace = hspace * (4 if N < 256 else 1)

    total_width = ((cols * N) + ((cols - 1) * wspace)) / dpi
    total_height = ((rows * N) + ((rows - 1) * hspace)) / dpi
    figsize = (total_width, total_height)

    fig, axarr = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    axarr = np.array(axarr).reshape(rows, cols)

    for idx, ax in enumerate(axarr.flat):
        if idx < num_arrays:
            cmap = cmaps[idx] if cmaps and idx < len(cmaps) else 'gray'
            ax.imshow(arrays[idx], cmap=cmap, vmin=0, vmax=1)
            if texts and idx < len(texts) and texts[idx]:
                text_set = texts[idx]
                text_position_set = text_positions[idx] if text_positions and idx < len(text_positions) else [
                                                                                                                 'top'] * len(
                    text_set)

                for text, text_position in zip(text_set, text_position_set):
                    if text_position == 'top':
                        ax.text(0.5, 1.01, text, transform=ax.transAxes, ha='center', va='bottom', size=font_size,
                                linespacing=line_height)
                    elif text_position == 'bottom':
                        ax.text(0.5, -0.01, text, transform=ax.transAxes, ha='center', va='top', size=font_size,
                                linespacing=line_height)
                    elif text_position == 'left':
                        ax.text(-0.01, 0.5, text, transform=ax.transAxes, ha='right', va='center', rotation=90,
                                size=font_size, linespacing=line_height)
                    elif text_position == 'right':
                        ax.text(1.01, 0.5, text, transform=ax.transAxes, ha='left', va='center', rotation=90,
                                size=font_size, linespacing=line_height)
        ax.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=wspace / N, hspace=hspace / N)

    if show:
        plt.show()

    for fmt in formats:
        savename = f"{filename}.{fmt}" if not filename.endswith(fmt) else filename
        plt.savefig(savename, format=fmt, bbox_inches='tight', pad_inches=0)
    plt.close()


# Function for plotting the dataset
def save_visualization(datafolder, simulation_class, start_frame_index=0,
                       end_frame_index=None, single=False, rows=1, space=10, save_path="", prefix=None, verbose=True):
    folders = [f for f in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, f))]

    sims = {}
    if simulation_class:
        sims[simulation_class.__name__] = simulation_class()

    progress_iterator = folders
    if verbose:
        progress_iterator = tqdm(progress_iterator, desc="Visualizing dataset")

    for unique_id in progress_iterator:
        files = [f for f in os.listdir(f'{datafolder}/{unique_id}') if f.endswith('.npy')]
        if len(files) < 1:
            continue

        if not simulation_class:
            simulation_name = unique_id.split('_')[1]
            if simulation_name not in sims:
                sims[simulation_name] = get_simulation_class(simulation_name)()
        else:
            simulation_name = simulation_class.__name__

        sim = sims[simulation_name]
        if single:
            for file in files:
                data = np.load(f'{datafolder}/{unique_id}/{file}')
                save_folder = f'{save_path}/{unique_id}'

                index = int(file.split('.')[0])
                os.makedirs(save_folder, exist_ok=True)

                file_name = f"{simulation_name}_{index}.png"
                out_path = os.path.join(save_folder, file_name)

                plot_arrays(data, out_path, cmaps=[sim.cmap])
        else:
            data = []
            for i in range(start_frame_index, end_frame_index if end_frame_index else len(files)):
                data.append(np.load(f'{datafolder}/{unique_id}/{i}.npy'))

            data = np.array(data)
            data = np.squeeze(data)

            save_folder = os.path.join(save_path, unique_id)
            os.makedirs(save_folder, exist_ok=True)

            file_name = f"{f'{prefix}_' if prefix else ''}{simulation_name}_{start_frame_index}_to_{end_frame_index if end_frame_index else len(files) - 1}.png"
            out_path = os.path.join(save_folder, file_name)

            plot_arrays(data, out_path, rows=rows, wspace=space, hspace=space,
                        cmaps=[sim.cmap for _ in range(data.shape[0])])


# Function for plotting the results
def save_result_visualization(datafolder, simulation_class, start_frame_index=0,
                              end_frame_index=None, single=False, rows=1, space=10, save_path="", prefix=None,
                              verbose=True):
    for result_data in ['inputs', 'trues', 'preds']:
        result_folder = os.path.join(datafolder, result_data)
        if not os.path.exists(result_folder):
            continue

        save_folder = os.path.join(save_path, result_data)
        save_visualization(result_folder, simulation_class, start_frame_index, end_frame_index, single, rows, space,
                           save_folder, prefix, verbose)


def calculate_tsse(actual, forecast):
    squared_errors = (actual - forecast) ** 2
    return np.sum(squared_errors)


# Function for plotting with the tsse metric
def plot_arrays_tsse(trues, preds, filename, cmap='coolwarm'):
    num_frames, height, width = trues.shape
    arrays = np.concatenate((trues, preds), axis=0)

    texts = []
    text_positions = []
    cmaps = []
    for idx in range(num_frames * 2):
        text = []
        text_position = []
        if idx < num_frames:
            if idx % 10 == 0:
                text.append("Ground Truth")
                text_position.append('left')
        else:
            if idx % 10 == 0:
                text.append("Prediction")
                text_position.append('left')

            tsse = calculate_tsse(trues[idx - num_frames], preds[idx - num_frames])
            text.append(f"TSSE: {tsse:.2f}")
            text_position.append('bottom')

        texts.append(text)
        text_positions.append(text_position)
        cmaps.append(cmap)

    rows = 2
    cols = num_frames

    plot_arrays(arrays, filename, rows=rows, cols=cols, texts=texts,
                text_positions=text_positions, font_size=18, cmaps=[cmaps])

# Function for plotting with the ssim metric
def plot_arrays_ssim(inputs, trues, preds, diff, filename, cmap='coolwarm', diff_cmap='gray', float_fmt='.5f'):
    num_frames, height, width = inputs.shape
    arrays = np.concatenate((inputs, trues, preds, diff), axis=0)

    print(inputs)

    texts = []
    text_positions = []
    cmaps = []
    for idx in range(num_frames * 4):
        text = []
        text_position = []
        if idx < num_frames:
            if idx % 10 == 0:
                text.append("Inputs")
                text_position.append('left')

            cmaps.append(cmap)
        elif idx < num_frames * 2:
            if idx % 10 == 0:
                text.append("Ground Truth")
                text_position.append('left')

            cmaps.append(cmap)
        elif idx < num_frames * 3:
            if idx % 10 == 0:
                text.append("Prediction")
                text_position.append('left')

            cmaps.append(cmap)
        else:
            if idx % 10 == 0:
                text.append("Difference")
                text_position.append('left')

            ssim_value = ssim(trues[idx - (num_frames * 3)], preds[idx - (num_frames * 3)])
            text.append(f"SSIM: {ssim_value:>{float_fmt}}")
            text_position.append('bottom')

            cmaps.append(diff_cmap)

        texts.append(text)
        text_positions.append(text_position)

    rows = 4
    cols = num_frames

    plot_arrays(arrays, filename, rows=rows, cols=cols, texts=texts,
                text_positions=text_positions, font_size=18, cmaps=cmaps)


def calculate_mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    non_zero_actual = actual != 0
    return np.mean(np.abs((actual[non_zero_actual] - forecast[non_zero_actual]) / actual[non_zero_actual])) * 100

# replaced with plot_arrays_metrics
# def show_video_line_metrics(metrics, trues, preds, ncols, vmax=1.0, vmin=0, cmap='gray', norm=None, cbar=False,
#                             format='png', out_path=None):
#     nrows = 2
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.25 * ncols, 6.5))
#     plt.subplots_adjust(wspace=0.1, hspace=0.1)
#
#     if len(trues.shape) > 3:
#         trues = trues.swapaxes(1, 2).swapaxes(2, 3)
#     if len(preds.shape) > 3:
#         preds = preds.swapaxes(1, 2).swapaxes(2, 3)
#
#     images = []
#     for t in range(ncols):
#         ax_true = axes[0, t]
#         true_img = trues[t]
#         im_true = ax_true.imshow(true_img, cmap=cmap, norm=norm)
#         ax_true.axis('off')
#         im_true.set_clim(vmin, vmax)
#
#         ax_pred = axes[1, t]
#         pred_img = preds[t]
#         im_pred = ax_pred.imshow(pred_img, cmap=cmap, norm=norm)
#         ax_pred.axis('off')
#         im_pred.set_clim(vmin, vmax)
#
#         images.append(im_true)
#         images.append(im_pred)
#
#         true = trues[t].squeeze()
#         pred = preds[t].squeeze()
#
#         ssim_value = ssim(true, pred)
#         mse_value = mean_squared_error(true, pred)
#         mae_value = np.mean(np.abs(true - pred))
#         mape_value = calculate_mape(true, pred)
#         tsse_value = calculate_tsse(true, pred)
#
#         text_y_start = -0.1
#         text_y_offset = -0.1
#
#         metrics_text = [
#             f"SSIM: {ssim_value:.5f}",
#             f"MSE: {mse_value:.5f}",
#             f"MAE: {mae_value:.5f}",
#             f"MAPE: {mape_value:.5f}",
#             f"TSSE: {tsse_value:.5f}",
#             f"MEAN SSIM: {metrics['SSIM'][t]:.5f}",
#             f"MEAN MSE: {metrics['MSE'][t]:.5f}",
#             f"MEAN MAE: {metrics['MAE'][t]:.5f}",
#             f"MEAN MAPE: {metrics['MAPE'][t]:.5f}",
#             f"MEAN TSSE: {metrics['TSSE'][t]:.5f}"
#         ]
#
#         for i, metric_text in enumerate(metrics_text):
#             ax_pred.text(0.5, text_y_start + i * text_y_offset, metric_text, size=12, ha="center",
#                          transform=ax_pred.transAxes)
#
#     if cbar and ncols > 1:
#         cbaxes = fig.add_axes([0.9, 0.15, 0.04 / ncols, 0.7 * nrows])
#         cbar = fig.colorbar(im_pred, ax=axes.ravel().tolist(), shrink=0.1, cax=cbaxes)
#
#     if out_path is not None:
#         fig.savefig(out_path, format=format, pad_inches=0, bbox_inches='tight')
#     plt.close()
#
# def show_video_line_ssim_comparison(inputs, trues, preds1, preds2, preds3, ncols, vmax=1.0, vmin=0, cmap='gray',
#                                     diff_cmap='gray', norm=None,
#                                     format='png', out_path=None):
#     nrows = 5
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5.3 * nrows))
#     plt.subplots_adjust(wspace=0.1, hspace=0.1)
#
#     if len(inputs.shape) > 3:
#         inputs = inputs.swapaxes(1, 2).swapaxes(2, 3)
#     if len(trues.shape) > 3:
#         trues = trues.swapaxes(1, 2).swapaxes(2, 3)
#     if len(preds1.shape) > 3:
#         preds1 = preds1.swapaxes(1, 2).swapaxes(2, 3)
#     if len(preds2.shape) > 3:
#         preds2 = preds2.swapaxes(1, 2).swapaxes(2, 3)
#     if len(preds3.shape) > 3:
#         preds3 = preds3.swapaxes(1, 2).swapaxes(2, 3)
#
#     for t in range(ncols):
#         ax_inputs = axes[0, t]
#         inputs_img = inputs[t]
#         im_inputs = ax_inputs.imshow(inputs_img, cmap=cmap, norm=norm)
#         ax_inputs.axis('off')
#         im_inputs.set_clim(vmin, vmax)
#
#         ax_pred1 = axes[1, t]
#         pred_img1 = preds1[t]
#         im_pred1 = ax_pred1.imshow(pred_img1, cmap=cmap, norm=norm)
#         ax_pred1.axis('off')
#         im_pred1.set_clim(vmin, vmax)
#
#         ax_pred2 = axes[2, t]
#         pred_img2 = preds2[t]
#         im_pred2 = ax_pred2.imshow(pred_img2, cmap=cmap, norm=norm)
#         ax_pred2.axis('off')
#         im_pred2.set_clim(vmin, vmax)
#
#         ax_pred3 = axes[3, t]
#         pred_img3 = preds3[t]
#         im_pred3 = ax_pred3.imshow(pred_img3, cmap=cmap, norm=norm)
#         ax_pred3.axis('off')
#         im_pred3.set_clim(vmin, vmax)
#
#         ax_true = axes[4, t]
#         true_img = trues[t]
#         im_true = ax_true.imshow(true_img, cmap=cmap, norm=norm)
#         ax_true.axis('off')
#         im_true.set_clim(vmin, vmax)
#
#         true = trues[t].squeeze()
#         pred1 = preds1[t].squeeze()
#         pred2 = preds2[t].squeeze()
#         pred3 = preds3[t].squeeze()
#
#         ssim_value1 = ssim(true, pred1)
#         ssim_value2 = ssim(true, pred2)
#         ssim_value3 = ssim(true, pred3)
#
#         if t == 0:
#             ax_inputs.text(x=-0.01, y=0.5, s='Inputs', rotation=90, size=32, va='center', ha='right',
#                            transform=ax_inputs.transAxes)
#             ax_pred1.text(x=-0.01, y=0.5, s='50s 50e', rotation=90, size=32, va='center', ha='right',
#                           transform=ax_pred1.transAxes)
#             ax_pred2.text(x=-0.01, y=0.5, s='250s 250e', rotation=90, size=32, va='center', ha='right',
#                           transform=ax_pred2.transAxes)
#             ax_pred3.text(x=-0.01, y=0.5, s='5000s 1000e', rotation=90, size=32, va='center', ha='right',
#                           transform=ax_pred3.transAxes)
#             ax_true.text(x=-0.01, y=0.5, s='Ground truth', rotation=90, size=32, va='center', ha='right',
#                          transform=ax_true.transAxes)
#
#         ax_pred1.text(0.5, -0.1, f"SSIM: {ssim_value1:.5f}", size=24, ha="center", transform=ax_pred1.transAxes)
#         ax_pred2.text(0.5, -0.1, f"SSIM: {ssim_value2:.5f}", size=24, ha="center", transform=ax_pred2.transAxes)
#         ax_pred3.text(0.5, -0.1, f"SSIM: {ssim_value3:.5f}", size=24, ha="center", transform=ax_pred3.transAxes)
#
#     if out_path is not None:
#         fig.savefig(out_path + ".png", format="png", pad_inches=0, bbox_inches='tight')
#         fig.savefig(out_path + ".pdf", format="pdf", pad_inches=0, bbox_inches='tight')
#     plt.close()
#
#
# def plot_metric(metric, metric_file, folder_path):
#     plt.figure()
#     plt.plot(metric)
#     plt.title(f"{metric_file.split('.')[0]} over Epochs")
#     plt.xlabel("Epochs")
#     plt.ylabel(metric_file.split('.')[0])
#     plt.grid(True)
#     plt.savefig(os.path.join(folder_path, f"{metric_file.replace('.npy', '.png')}"))
#
#
# def plot_combined_loss(train_loss, vali_loss, folder_path):
#     plt.figure()
#     plt.plot(train_loss, label='Train Loss')
#     plt.plot(vali_loss, label='Validation Loss')
#     plt.title("Training and Validation Loss over Epochs")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(folder_path, 'loss.png'))
#
#
# def calculate_column_metrics(trues, preds):
#     n_columns = trues.shape[1]
#     metrics = {'SSIM': [], 'MSE': [], 'MAE': [], 'MAPE': [], 'TSSE': []}
#
#     for column in range(n_columns):
#         ssim_values, mse_values, mae_values, mape_values, tsse_values = [], [], [], [], []
#         for row in range(trues.shape[0]):
#             true_img = trues[row, column].squeeze()
#             pred_img = preds[row, column].squeeze()
#
#             ssim_values.append(ssim(true_img, pred_img))
#             mse_values.append(mean_squared_error(true_img, pred_img))
#             mae_values.append(np.mean(np.abs(true_img - pred_img)))
#             mape_values.append(calculate_mape(true_img, pred_img))
#             tsse_values.append(calculate_tsse(true_img, pred_img))
#
#         metrics['SSIM'].append(np.mean(ssim_values))
#         metrics['MSE'].append(np.mean(mse_values))
#         metrics['MAE'].append(np.mean(mae_values))
#         metrics['MAPE'].append(np.mean(mape_values))
#         metrics['TSSE'].append(np.mean(tsse_values))
#
#     return metrics
#
#
# def save_metrics_statistics(metrics, output_path):
#     with open(output_path + '.txt', 'w') as f:
#         for metric_name, column_values in metrics.items():
#             for column_index, value in enumerate(column_values):
#                 # Writing metric statistics per column
#                 f.write(f"{metric_name} - Column {column_index + 1}: {value:.10f}\n")
#
#     for metric_name, column_values in metrics.items():
#         plt.figure(figsize=(8, 6))
#         plt.plot(range(1, len(column_values) + 1), column_values, marker='o', linestyle='-', color='b')
#         plt.xlabel('Column Number')
#         plt.ylabel(metric_name)
#         plt.title(f'{metric_name} Averages per Column')
#         plt.grid(True)
#
#         plot_output_path = output_path + f'_{metric_name}.png'
#         plt.savefig(plot_output_path)
#         plt.close()


# def save_result_visualizations(save_folder, simulation, normalized=True, result_suffix=""):
#     inputs = np.load(f'{save_folder}/inputs{result_suffix}.npy')
#     trues = np.load(f'{save_folder}/trues{result_suffix}.npy')
#     preds = np.load(f'{save_folder}/preds{result_suffix}.npy')
#
#     vmax = (1 if normalized else simulation.vmax)
#     prefix = simulation.__name__.lower()
#
#     pre_seq_length, aft_seq_length = inputs.shape[1], trues.shape[1]
#
#     example_idx = 0
#
#     show_video_line(inputs[example_idx], ncols=pre_seq_length, vmax=vmax, vmin=0, cbar=False, format='png',
#                     cmap=simulation.cmap,
#                     out_path=f'{save_folder}/{prefix}_input{result_suffix}.png')
#     show_video_line(preds[example_idx], ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False, format='png',
#                     cmap=simulation.cmap,
#                     out_path=f'{save_folder}/{prefix}_pred{result_suffix}.png')
#     show_video_line(trues[example_idx], ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False, format='png',
#                     cmap=simulation.cmap,
#                     out_path=f'{save_folder}/{prefix}_true{result_suffix}.png')
#
#     diff = np.abs(preds[example_idx] - trues[example_idx])
#     show_video_line(diff, ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False, format='png', cmap=simulation.diff_cmap,
#                     out_path=f'{save_folder}/{prefix}_diff{result_suffix}.png')
#
#     show_video_gif_multiple(inputs[example_idx], trues[example_idx], preds[example_idx], vmax=vmax, vmin=0,
#                             cmap=simulation.cmap,
#                             out_path=f'{save_folder}/{prefix}{result_suffix}.gif')
#
#     metrics = calculate_column_metrics(trues, preds)
#     output_path = f'{save_folder}/metrics_statistics'
#     save_metrics_statistics(metrics, output_path)
#
#     for idx in range(0, min(trues.shape[0], 5)):
#         show_video_line_metrics(metrics, trues[idx], preds[idx], ncols=aft_seq_length, vmax=vmax, vmin=0, cbar=False,
#                                 format='png',
#                                 cmap=simulation.cmap,
#                                 out_path=f'{save_folder}/{prefix}_metrics_{idx}.png')
#         show_video_line_ssim(inputs[idx], trues[idx], preds[idx], diff, ncols=aft_seq_length, vmax=vmax, vmin=0,
#                              cbar=False,
#                              format='png',
#                              cmap=simulation.cmap,
#                              diff_cmap=simulation.diff_cmap,
#                              out_path=f'{save_folder}/{prefix}_ssim_{idx}.png')
#
#     metric_files = ['mse.npy', 'mae.npy', 'lr.npy', 'train_loss.npy', 'vali_loss.npy']
#
#     train_loss, vali_loss = None, None
#     for metric_file in metric_files:
#         metric_path = os.path.join(save_folder, metric_file)
#         if os.path.exists(metric_path):
#             metric = np.load(metric_path)
#             plot_metric(metric, metric_file, save_folder)
#             if metric_file == 'train_loss.npy':
#                 train_loss = metric
#             if metric_file == 'vali_loss.npy':
#                 vali_loss = metric
#         else:
#             print(f"Metric file {metric_file} not found in {save_folder}")
#
#     # Plot combined train and validation loss
#     if train_loss is not None and vali_loss is not None:
#         plot_combined_loss(train_loss, vali_loss, save_folder)


# def save_result_visualizations_comparison(save_folder1, save_folder2, save_folder3, simulation, normalized=True,
#                                           result_suffix=""):
#     inputs = np.load(f'{save_folder1}/inputs{result_suffix}.npy')
#     trues = np.load(f'{save_folder1}/trues{result_suffix}.npy')
#     preds1 = np.load(f'{save_folder1}/preds{result_suffix}.npy')
#     preds2 = np.load(f'{save_folder2}/preds{result_suffix}.npy')
#     preds3 = np.load(f'{save_folder3}/preds{result_suffix}.npy')
#
#     vmax = (1 if normalized else simulation.vmax)
#     prefix = simulation.__name__.lower()
#
#     pre_seq_length, aft_seq_length = inputs.shape[1], trues.shape[1]
#
#     for idx in range(0, min(trues.shape[0], 5)):
#         show_video_line_ssim_comparison(inputs[idx], trues[idx], preds1[idx], preds2[idx], preds3[idx],
#                                         ncols=aft_seq_length, vmax=vmax, vmin=0,
#                                         cmap=simulation.cmap,
#                                         diff_cmap=simulation.diff_cmap,
#                                         format='png',
#                                         out_path=f'./work_dirs/{prefix}_ssim_comparison_{idx}')

# def show_video_gif_single(data, vmax=0.6, vmin=0.0, cmap='gray', norm=None, out_path=None, use_rgb=False):
#     """generate gif with a video sequence"""
#
#     def swap_axes(x):
#         if len(x.shape) > 3:
#             return x.swapaxes(1, 2).swapaxes(2, 3)
#         else:
#             return x
#
#     data = swap_axes(data)
#     images = []
#     for i in range(data.shape[0]):
#         fig, ax = plt.subplots()
#         if use_rgb:
#             img = ax.imshow(cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB))
#         else:
#             img = ax.imshow(data[i], cmap=cmap, norm=norm)
#         img.set_clim(vmin, vmax)
#         ax.axis('off')
#         plt.savefig('./tmp.png', bbox_inches='tight', format='png')
#         images.append(imageio.imread('./tmp.png'))
#         plt.close()
#     os.remove('./tmp.png')
#
#     if out_path is not None:
#         if not out_path.endswith('gif'):
#             out_path += '.gif'
#         imageio.mimsave(out_path, images)


# def show_video_line(data, ncols, vmax=0.6, vmin=0.0, cmap='gray', norm=None, cbar=False, format='png',
#                          out_path=None, use_rgb=False):
#     """generate images with a video sequence"""
#     nrows = 1
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.25 * ncols, 6.5))
#     plt.subplots_adjust(wspace=0.1, hspace=0.1)
#
#     if len(data.shape) > 3:
#         data = data.swapaxes(1, 2).swapaxes(2, 3)
#
#     images = []
#     for t in range(ncols):
#         ax_true = axes[0, t]
#         true_img = cv2.cvtColor(data[t], cv2.COLOR_BGR2RGB) if use_rgb else data[t]
#         im_true = ax_true.imshow(true_img, cmap=cmap, norm=norm)
#         ax_true.axis('off')
#         im_true.set_clim(vmin, vmax)
#
#         images.append(im_true)
#
#     if cbar and ncols > 1:
#         cbaxes = fig.add_axes([0.9, 0.15, 0.04 / ncols, 0.7 * nrows])
#         cbar = fig.colorbar(im_true, ax=axes.ravel().tolist(), shrink=0.1, cax=cbaxes)
#
#     # plt.show()
#     if out_path is not None:
#         fig.savefig(out_path, format=format, pad_inches=0, bbox_inches='tight')
#     plt.close()
