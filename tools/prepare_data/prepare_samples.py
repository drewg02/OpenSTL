import os
import numpy as np

from openstl.utils import create_parser

from openstl.simulation.simulations import simulations
from openstl.simulation.preparation import normalize_data_min_max
from openstl.simulation.utils import get_simulation_class

# Create a parser with additional arguments specific to dataset preparation.
def create_local_parser():
    parser = create_parser()
    parser.description = "Prepare a dataset for a simulation."

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                        help='Determines the simulation type.', required=True)

    parser.add_argument('--offset', type=int, default=0,
                        help='Specifies the offset to remove from the start of each sample.')

    parser.add_argument('--normalize', action='store_true',
                        help='Determines whether to normalize the data.')

    # Add arguments so that we can split each sample into multiple samples
    # For example if we (1000, 100, 1, 64, 64) that means we have 1000 samples, each sample has 100 time steps, 1 channel, 64x64 image
    # If we want to we can pull 5 random samples of length 20 from each sample, we can set pre_seq_length=10, aft_seq_length=10
    # This will give us 5*1000 samples of length 20, 1 channel, 64x64 image
    # So we need to add 1 new argument that is an int, by default it is None
    # This new argument will be the number of random samples to pull from each sample
    # It will use the pre_seq_length, aft_seq_length, and total_length to determine the length of each new sample
    # We don't need to add an argument for pre_seq_length or aft_seq_length, they are predefined
    # We can add a new argument to the parser, called num_random_samples
    # We want this to be done after everything else
    # parser.add_argument('--num_random_samples', type=int, default=None, help='Specifies the number of random samples to pull from each sample.')

    parser.add_argument('--datafolder_in', type=str, required=True,
                        help='Specifies the input data file path.')
    parser.add_argument('--datafolder_out', type=str, required=True,
                        help='Specifies the output data file path.')

    return parser

def main():
    args = create_local_parser().parse_args()  # Parse the command line arguments.

    # Default output file to input file if not specified.
    if not args.datafolder_out:
        args.datafolder_out = args.datafolder_in

    # Get the simulation class.
    simulation_class = get_simulation_class(args.simulation)
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {args.simulation}")

    simulation = simulation_class(args)  # Initialize the simulation class.

    for filename in os.listdir(args.datafolder_in):
        if filename.endswith('.npy'):
            file_path = os.path.join(args.datafolder_in, filename)
            data = np.load(file_path)

            # Apply offset and total_length adjustments
            if args.offset > 0:
                data = data[args.offset:]

            total_length = args.total_length
            if total_length is None and args.pre_seq_length and args.aft_seq_length:
                total_length = args.pre_seq_length + args.aft_seq_length

            if total_length is not None and total_length != data.shape[1]:
                data = data[:total_length]

            if args.normalize:
                data = normalize_data_min_max(data, simulation.vmin, simulation.vmax)

            save_path = os.path.join(args.datafolder_out, filename)
            np.save(save_path, data)

if __name__ == '__main__':
    main()
