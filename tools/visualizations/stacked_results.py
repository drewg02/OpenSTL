import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import time


plt.rcParams.update({'font.size': 12})  # Adjust this value as needed

def show_video_line_stacked(trues, preds_dict, ncols, vmax=1.0, vmin=0, cmap='gray', norm=None, cbar=False, format='png', out_path=None, use_rgb=False):
    nrows = 1 + len(preds_dict)  # The number of rows needed
    fig_width = 3.25 * ncols  # Width of each column
    fig_height_per_row = 3  # Height of each row, adjust as needed for better visualization
    
    fig_height = fig_height_per_row * nrows  # Total height of the figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the spacing between images if necessary
    
    if len(trues.shape) > 3:
        trues = trues.swapaxes(1,2).swapaxes(2,3)
    
    for key in preds_dict.keys():
        if len(preds_dict[key].shape) > 3:
            preds_dict[key] = preds_dict[key].swapaxes(1,2).swapaxes(2,3)
    
    for t in range(ncols):
        ax_true = axes[0, t]
        true_img = trues[t]
        im_true = ax_true.imshow(true_img, cmap=cmap, norm=norm)
        ax_true.axis('on')  # Turn the axis on
        ax_true.set_xticks([])  # Remove x-axis tick marks
        ax_true.set_yticks([])  # Remove y-axis tick marks
        if t == 0:  # Only add y label to the first column
            ax_true.set_ylabel('Ground truth')
        im_true.set_clim(vmin, vmax)

        i = 0  # Counter for positioning in subplot
        for key in preds_dict:
            ax_pred = axes[i + 1, t]
            pred_img = preds_dict[key][t]
            im_pred = ax_pred.imshow(pred_img, cmap=cmap, norm=norm)
            ax_pred.axis('on')  # Turn the axis on
            ax_pred.set_xticks([])  # Initially remove x-axis tick marks
            ax_pred.set_yticks([])  # Remove y-axis tick marks
            if t == 0:  # Only add y label to the first column
                ax_pred.set_ylabel(f'{key} plates')
            im_pred.set_clim(vmin, vmax)
            i += 1
    
    # Set x-axis labels for the last row of the figure to number 1 through 10
    for ax, label in zip(axes[-1], range(1, ncols + 1)):
        ax.set_xlabel(str(label))

    # Adjust this value to increase/decrease the padding around the figure when saving
    extra_padding = 0.25  # Amount of extra padding around the figure in inches

    if out_path is not None:
        fig.savefig(out_path, format=format, pad_inches=extra_padding, bbox_inches='tight')

    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph the video line results for multiple plate sizes")
    parser.add_argument("save_path", type=str, help="Save path")
    parser.add_argument("ex_name", type=str, help="Ex name")
    parser.add_argument("plate_size", type=int, help="Plate size")
    parser.add_argument("plate_sizes", nargs='+', type=int, help="Plate sizes")
    args = parser.parse_args()
    
    idx = 0
    trues = np.load(f'./work_dirs/{args.ex_name}/saved/trues.npy')[idx]
    
    preds_dict = {}  # Initialize preds_dict as a dictionary
    for plate_size in args.plate_sizes:
        ex_name = args.ex_name.replace(f'{args.plate_size}plates', f'{plate_size}plates')
        print(f'Starting run for ex {ex_name}')
        start_time = time.time()
        
        save_folder = f'./work_dirs/{ex_name}/saved'
        preds = np.load(f'{save_folder}/preds.npy')
        preds_dict[plate_size] = preds[idx]  # Add preds to preds_dict with plate_size as key
        
        print(f'Finished run for ex {ex_name}, took {time.time() - start_time} seconds')
    
    show_video_line_stacked(trues, preds_dict, ncols=10, vmax=0.6, cbar=False, format='png', cmap='coolwarm', out_path=args.save_path)
