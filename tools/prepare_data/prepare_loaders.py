import os
import json

from argparse import ArgumentParser as ArgParser

from openstl.simulation.preparation import load_files, train_val_test_split_files

def create_local_parser():
    parser = ArgParser()
    parser.description = "Select a random subset of files, then split into train, validation, and test sets."

    parser.add_argument('--num_samples', type=int, help='Total number of files to randomly select from the dataset.')

    parser.add_argument('--sample_start_index', type=int, default=0, help='Specifies the starting index of the sequence to be used for input, -1 for random.')
    parser.add_argument('--total_length', type=int, default=0, help='Specifies the total length to use for the sequence, 0 to use full sequence.')

    parser.add_argument('--train_ratio', type=float, default=0.7, help='Specifies the training data ratio.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Specifies the validation data ratio.')

    parser.add_argument('--datafolder', type=str, required=True, help='Path to the data folder.')

    return parser

def main():
    args = create_local_parser().parse_args()

    files = load_files(args.datafolder, args.num_samples, args.sample_start_index, args.total_length)
    train_val_test_splits = train_val_test_split_files(files, args.train_ratio, args.val_ratio)

    if not os.path.exists(args.datafolder):
        raise FileNotFoundError(f"Data folder in path {args.datafolder} does not exist.")

    # Save each split into its own file
    with open(os.path.join(args.datafolder, f"loader.json"), 'w') as f:
        json.dump(train_val_test_splits, f, indent=4)

    print(f"File splits saved to {args.datafolder}")

if __name__ == '__main__':
    main()