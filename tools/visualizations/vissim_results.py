import argparse

from openstl.simulation.simulations import simulations
from openstl.simulation.utils import get_simulation_class
from openstl.simulation.visualization import save_result_visualization


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create visualizations for each experiment.")

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                        help='Determines the simulation type.', required=False)

    parser.add_argument('--start_frame_index', type=int, default=0,
                        help='Specifies the start index of the frames for visualization.')
    parser.add_argument('--end_frame_index', type=int, default=None,
                        help='Specifies the end index of the frames for visualization.')

    parser.add_argument('--output_single_images', action='store_true',
                        help='If set, output single images for each frame. Otherwise, output a concatenated line of images.')
    parser.add_argument('--rows', type=int, default=1,
                        help='Specifies the number of rows for the concatenated images.')
    parser.add_argument('--space', type=int, default=10,
                        help='Specifies the space between the images in the concatenated output.')

    parser.add_argument('--datafolder', type=str, required=True,
                        help='Specifies the input data file path.')
    parser.add_argument('--save_path', type=str,
                        help='Specifies the folder path where the visualizations will be saved.', required=True)

    args = parser.parse_args()

    if args.simulation:
        simulation_class = get_simulation_class(args.simulation)
        if not simulation_class:
            raise ValueError(f"Invalid simulation: {args.simulation}")
    else:
        simulation_class = None

    save_result_visualization(args.datafolder, simulation_class,
                              args.start_frame_index, args.end_frame_index, args.output_single_images, args.rows,
                              args.space, args.save_path)


if __name__ == "__main__":
    main()
