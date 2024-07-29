import time
import os

from argparse import ArgumentParser as ArgParser
from simvp_standalone.experiment_recorder import generate_experiment_record, save_experiment_record

from simvp_standalone.generation import create_samples


# Create a parser with additional arguments specific to dataset generation.
def create_local_parser():
    parser = ArgParser()

    parser.description = "Generate a dataset for a simulation."

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
    args_dict = vars(args)

    start_time = time.time()  # Record the start time.
    # Generate samples based on the provided parameters.
    create_samples(args.total_sample_length, args.datafolder, args.normalize, args_dict, verbose=True)

    elapsed = time.time() - start_time  # Calculate the elapsed time.

    samples_arguments = {
        'increment': args.increment,
        'total_sample_length': args.total_sample_length,
        'normalize': args.normalize
    }

    record = generate_experiment_record(**samples_arguments)
    save_experiment_record(record, os.path.join(args.datafolder, f"{record['id']}_samples.json"))

    print(f"Generating samples took {elapsed} seconds.")  # Print the time taken to generate samples.

if __name__ == '__main__':
    main()
