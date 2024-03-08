import os
import time
import numpy as np

from openstl.utils import create_parser

from openstl.simulation.simulations import simulations
from openstl.simulation.generation import ArrayType, create_samples
from openstl.simulation.utils import save_data, get_simulation_class, get_seq_lengths


# Create a parser with additional arguments specific to dataset generation.
def create_local_parser():
    parser = create_parser()
    parser.description = "Generate a dataset for a simulation."

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                        help='Determines the simulation type.', required=True)

    parser.add_argument('--num_samples', type=int,
                        help='Specifies the number of samples.', required=True)
    parser.add_argument('--image_height', type=int, default=64,
                        help='Specifies the image height.')
    parser.add_argument('--image_width', type=int, default=64,
                        help='Specifies the image width.')

    parser.add_argument('--array_type', type=ArrayType, default=ArrayType.RANDOM,
                        help='Defines the array type.')
    parser.add_argument('--chance', type=float, default=0.1,
                        help='Sets the chance parameter.')
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--static_cells_random', action='store_true')
    parser.add_argument('--dynamic_cells_random', action='store_true')
    parser.add_argument('--increment', type=int, default=5,
                        help="Only applies to Boiling simulation, sets the increment value.")

    parser.add_argument('--datafile_out', type=str, required=True,
                        help='Specifies the data file path.')
    parser.add_argument('--preserve_datafile', action='store_true',
                        help='If set, the script will not overwrite the existing data file.')

    return parser


def main():
    args = create_local_parser().parse_args()  # Parse the command line arguments.

    # Check if the data file should be preserved and exists.
    if args.preserve_datafile and os.path.exists(args.datafile_out):
        print(f"File {args.datafile_out} already exists.")
        return

    # Get the simulation class.
    simulation_class = get_simulation_class(args.simulation)
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {args.simulation}")

    simulation = simulation_class(args)  # Initialize the simulation class.

    # Get sequence lengths for the simulation.
    pre_seq_length, aft_seq_length, total_length = get_seq_lengths(args)
    if total_length is None:
        raise ValueError("Must specify --total_length or --pre_seq_length and --aft_seq_length")

    start_time = time.time()  # Record the start time.
    # Generate samples based on the provided parameters.
    samples = create_samples(args.image_height, args.image_width, args.num_samples, total_length, simulation,
                             args.array_type, args.thickness, args.chance, args.static_cells_random,
                             args.dynamic_cells_random, verbose=True)
    # Reshape the samples into the correct format.
    samples = np.reshape(samples, (args.num_samples, total_length, 1, args.image_height, args.image_width))

    elapsed = time.time() - start_time  # Calculate the elapsed time.
    print(f"Generating {len(samples)} samples took {elapsed} seconds.")  # Print the time taken to generate samples.

    save_data(samples, args.datafile_out)  # Save the generated samples to the specified file.


if __name__ == '__main__':
    main()
