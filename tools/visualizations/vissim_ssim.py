import os
import argparse

import numpy as np

from openstl.simulation.simulations import simulations
from openstl.simulation.utils import get_simulation_class
from openstl.simulation.visualization import pick_data, load_data, plot_arrays_ssim


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create visualizations for each experiment.")

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                        help='Determines the simulation type.', required=True)

    parser.add_argument('--unique_ids', type=str, nargs='+',
                        help='Specifies the unique ids of the experiments to visualize.')
    parser.add_argument('--limit', type=int,
                        help='Specifies the number of experiments to visualize.', default=25)

    parser.add_argument('--datafolder', type=str, required=True,
                        help='Specifies the input data file path.')
    parser.add_argument('--save_path', type=str,
                        help='Specifies the folder path where the visualization will be saved.', required=True)

    args = parser.parse_args()

    simulation_class = get_simulation_class(args.simulation)
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {args.simulation}")

    if args.unique_ids:
        unique_ids = args.unique_ids
    else:
        unique_ids = pick_data(os.path.join(args.datafolder, 'inputs'), args.limit)

    inputs = load_data(os.path.join(args.datafolder, 'inputs'), unique_ids)
    trues = load_data(os.path.join(args.datafolder, 'trues'), unique_ids)
    preds = load_data(os.path.join(args.datafolder, 'preds'), unique_ids)
    diff = np.abs(trues - preds)

    for i in range(min(inputs.shape[0], trues.shape[0], preds.shape[0], 25)):
        filename = os.path.join(args.save_path, f'{simulation_class.__name__.lower()}_ssim_{i}.png')
        plot_arrays_ssim(inputs[i], trues[i], preds[i], diff[i], filename, cmap=simulation_class.cmap,
                         diff_cmap=simulation_class.diff_cmap)


if __name__ == "__main__":
    main()
