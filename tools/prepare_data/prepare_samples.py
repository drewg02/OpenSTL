import os
from openstl.utils import create_parser
from openstl.simulation.simulations import simulations
from openstl.simulation.preparation import normalize_data_min_max, train_val_test_split_data, X_Y_split_data
from openstl.simulation.utils import load_data, save_data, get_simulation_class, get_seq_lengths

def create_local_parser():
    parser = create_parser()

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                        help='Select the type of simulation that was used.', required=True)

    parser.add_argument('--num_samples', type=int, required=False)
    parser.add_argument('--offset', type=int, default=0)

    parser.add_argument('--normalize', type=bool, default=False)

    parser.add_argument('--train_val_test_split', type=bool, default=False)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)

    parser.add_argument('--X_Y_split', type=bool, default=False)

    parser.add_argument('--datafile_in', type=str, required=True)
    parser.add_argument('--datafile_out', type=str)
    parser.add_argument('--overwrite_datafile', type=bool, default=False)

    return parser

def main():
    args = create_local_parser().parse_args()

    if not args.datafile_out:
        args.datafile_out = args.datafile_in

    if not args.overwrite_datafile and os.path.exists(args.datafile_out):
        print(f"File {args.datafile_out} already exists. Set --overwrite_datafile' to True to overwrite.")
        return

    simulation_class = get_simulation_class(args.simulation)
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {args.simulation}")

    simulation = simulation_class(args)

    if args.train_val_test_split and args.train_ratio + args.val_ratio >= 1.0:
            raise ValueError(f"train_ratio ({args.train_ratio}) + val_ratio ({args.val_ratio}) must be less than 1.0")

    dataset = load_data(args.datafile_in)
    is_dict = isinstance(dataset, dict)

    pre_seq_length, aft_seq_length = None, None
    total_length = (dataset[next(iter(dataset.keys()))] if is_dict else dataset).shape[1]

    if args.pre_seq_length or args.aft_seq_length:
        pre_seq_length, aft_seq_length, total_length = get_seq_lengths(args) or (None, None, total_length)

    pre_seq_length = pre_seq_length or (total_length - aft_seq_length if aft_seq_length else None)

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

    if args.X_Y_split:
        if not pre_seq_length:
            raise ValueError("Must specify --pre_seq_length or --aft_seq_length to split data into X and Y")
        dataset = X_Y_split_data(dataset, pre_seq_length)

    save_data(dataset, args.datafile_out)


if __name__ == '__main__':
    main()
