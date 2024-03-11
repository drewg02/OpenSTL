import argparse
import time

from openstl.simulation.visualization import save_dataset_visualization
from openstl.simulation.simulations import simulations
from openstl.simulation.utils import load_data, get_simulation_class


# Create a parser with additional arguments specific to dataset visualization.
def main():
    parser = argparse.ArgumentParser(description="Visualize a dataset for a simulation.")

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                        help='Determines the simulation type.', required=True)

    parser.add_argument('--start_sample_index', type=int, default=0,
                        help='Defines the start index of the samples to visualize.')
    parser.add_argument('--end_sample_index', type=int, default=1,
                        help='Defines the end index of the samples to visualize.')

    parser.add_argument('--start_frame_index', type=int, default=0,
                        help='Specifies the start index of the frames for visualization.')
    parser.add_argument('--end_frame_index', type=int, default=None,
                        help='Specifies the end index of the frames for visualization.')

    parser.add_argument('--output_single_images', action='store_true',
                        help='If set, output single images for each frame. Otherwise, output a concatenated line of images.')
    parser.add_argument('--normalized', action='store_true',
                        help='If set, the input data is expected to be normalized.')

    parser.add_argument('--datafile_in', type=str, required=True,
                        help='Specifies the input data file path.')
    parser.add_argument('--save_path', type=str,
                        help='Specifies the folder path where the visualizations will be saved.', required=True)

    args = parser.parse_args()

    # Get the simulation class.
    simulation_class = get_simulation_class(args.simulation)
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {args.simulation}")

    dataset = load_data(args.datafile_in)  # Load the dataset.

    # Begin the visualization process, tracking the time taken.
    print(f'Starting visualization of dataset {args.datafile_in}')
    start_time = time.time()

    # Generate visualizations for the specified dataset range and save them to the designated path.
    save_dataset_visualization(dataset, simulation_class, args.start_sample_index, args.end_sample_index,
                               args.start_frame_index, args.end_frame_index, args.output_single_images, args.save_path,
                               args.normalized)

    # Print out the completion message and the time taken for the visualization process.
    print(f'Finished visualization of dataset {args.datafile_in}, took {time.time() - start_time} seconds')


if __name__ == "__main__":
    main()
