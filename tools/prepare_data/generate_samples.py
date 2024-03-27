import os
import time

from openstl.utils import create_parser

from openstl.simulation.simulations import simulations
from openstl.simulation.generation import create_samples
from openstl.simulation.utils import get_simulation_class, get_seq_lengths


# Create a parser with additional arguments specific to dataset generation.
def create_local_parser():
    parser = create_parser()
    parser.description = "Generate a dataset for a simulation."

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                        help='Determines the simulation type.', required=True)

    parser.add_argument('--num_samples', type=int,
                        help='Specifies the number of samples.', required=True)

    parser.add_argument('--increment', type=int, default=5,
                        help="Only applies to Boiling simulation, sets the increment value.")

    parser.add_argument('--datafolder_in', type=str, required=True,
                        help='Specifies the data folder in path.')
    parser.add_argument('--datafolder_out', type=str, required=True,
                        help='Specifies the data folder out path.')

    return parser


def main():
    args = create_local_parser().parse_args()  # Parse the command line arguments.

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
    create_samples(args.num_samples, total_length, simulation, args.datafolder_in, args.datafolder_out, verbose=True)

    elapsed = time.time() - start_time  # Calculate the elapsed time.
    print(f"Generating {args.num_samples} samples took {elapsed} seconds.")  # Print the time taken to generate samples.


if __name__ == '__main__':
    main()
