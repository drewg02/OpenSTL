import time
import os
import json

from argparse import ArgumentParser as ArgParser
from experiment_recorder import generate_experiment_record, save_experiment_record

from openstl.simulation.simulations import simulations
from openstl.simulation.generation import create_samples
from openstl.simulation.utils import get_simulation_class
from openstl.simulation.preparation import normalize_samples


# Create a parser with additional arguments specific to dataset generation.
def create_local_parser():
    parser = ArgParser()

    parser.description = "Generate a dataset for a simulation."

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                        help='Determines the simulation type.', required=True)

    parser.add_argument('--increment', type=int, default=5,
                        help="Only applies to Boiling simulation, sets the increment value.")
    parser.add_argument('--total_sample_length', type=int, help='Number of total iterations in the sequence including the initial state.', required=True)

    parser.add_argument('--normalize', action='store_true',
                        help='Determines whether to normalize the data.')

    parser.add_argument('--datafolder', type=str, required=True,
                        help='Specifies the data folder in path.')

    return parser


def main():
    args = create_local_parser().parse_args()  # Parse the command line arguments.

    # Get the simulation class.
    simulation_class = get_simulation_class(args.simulation)
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {args.simulation}")

    simulation = simulation_class(args)  # Initialize the simulation class.

    if not os.path.exists(args.datafolder):
        raise FileNotFoundError(f"Data folder in path {args.datafolder} does not exist.")

    start_time = time.time()  # Record the start time.
    # Generate samples based on the provided parameters.
    create_samples(args.total_sample_length, simulation, args.datafolder, verbose=True)
    if args.normalize:
        normalize_samples(args.datafolder, simulation.vmin, simulation.vmax)  # Normalize the samples.

    elapsed = time.time() - start_time  # Calculate the elapsed time.

    samples_arguments = {
        'simulation': args.simulation,
        'increment': args.increment,
        'total_sample_length': args.total_sample_length,
        'normalize': args.normalize
    }

    record = generate_experiment_record(**samples_arguments)
    save_experiment_record(record, os.path.join(args.datafolder, f"{record['id']}_samples.json"))

    print(f"Generating samples took {elapsed} seconds.")  # Print the time taken to generate samples.

if __name__ == '__main__':
    main()
