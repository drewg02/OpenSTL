import os

from openstl.utils import create_parser

from openstl.simulation.simulations import simulations
from openstl.simulation.preparation import normalize_data_min_max, train_val_test_split_data, X_Y_split_data
from openstl.simulation.utils import load_data, save_data, get_simulation_class, get_seq_lengths

# Create a parser with additional arguments specific to dataset preparation.
def create_local_parser():
    parser = create_parser()
    parser.description = "Prepare a dataset for a simulation."

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                        help='Determines the simulation type.', required=True)

    parser.add_argument('--num_samples', type=int,
                        help='Specifies the number of samples to trim the input dataset to.', required=True)
    parser.add_argument('--offset', type=int, default=0,
                        help='Specifies the offset to remove from the start of each sample.')

    parser.add_argument('--normalize', action='store_true',
                        help='Determines whether to normalize the data.')

    parser.add_argument('--train_val_test_split', action='store_true',
                        help='Determines if data should be split into training, validation, and test sets.')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Specifies the training data ratio.')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Specifies the validation data ratio.')

    parser.add_argument('--X_Y_split', action='store_true',
                        help='Determines if data should be split into input (X) and output (Y).')

    parser.add_argument('--datafile_in', type=str, required=True,
                        help='Specifies the input data file path.')
    parser.add_argument('--datafile_out', type=str, required=True,
                        help='Specifies the output data file path.')
    parser.add_argument('--preserve_datafile', action='store_true',
                        help='Prevents overwriting the existing data file if set.')

    return parser

def main():
    args = create_local_parser().parse_args()  # Parse the command line arguments.

    # Default output file to input file if not specified.
    if not args.datafile_out:
        args.datafile_out = args.datafile_in

    # Check if the output file should be preserved and exists.
    if args.preserve_datafile and os.path.exists(args.datafile_out):
        print(f"File {args.datafile_out} already exists.")
        return

    # Get the simulation class.
    simulation_class = get_simulation_class(args.simulation)
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {args.simulation}")

    simulation = simulation_class(args)  # Initialize the simulation class.

    # Validate train and validation split ratios.
    if args.train_val_test_split and args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({args.train_ratio}) + val_ratio ({args.val_ratio}) must be less than 1.0")

    dataset = load_data(args.datafile_in)  # Load the dataset.
    is_dict = isinstance(dataset, dict)  # Check if the dataset is a dictionary.

    # Determine the sequence lengths for processing.
    pre_seq_length, aft_seq_length = None, None
    total_length = (dataset[next(iter(dataset.keys()))] if is_dict else dataset).shape[1]

    if args.pre_seq_length or args.aft_seq_length:
        pre_seq_length, aft_seq_length, total_length = get_seq_lengths(args) or (None, None, total_length)

    pre_seq_length = pre_seq_length or (total_length - aft_seq_length if aft_seq_length else None)

    # Process dataset based on provided arguments.
    if not is_dict:
        if args.num_samples:
            dataset = dataset[:args.num_samples]
        if total_length != dataset.shape[1]:
            dataset = dataset[:, :total_length, ...]
        if args.offset:
            dataset = dataset[:, args.offset:, ...]
        if args.normalize:
            dataset = normalize_data_min_max(dataset, simulation.vmin, simulation.vmax)
        if args.train_val_test_split:
            dataset = train_val_test_split_data(dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # Split data into input (X) and output (Y) if required.
    if args.X_Y_split:
        if not pre_seq_length:
            raise ValueError("Must specify --pre_seq_length or --aft_seq_length to split data into X and Y")
        dataset = X_Y_split_data(dataset, pre_seq_length)

    save_data(dataset, args.datafile_out)  # Save the processed data.

if __name__ == '__main__':
    main()
