import os
import json

from openstl.simulation.preparation import train_val_test_split_files
from openstl.utils import create_parser

def create_local_parser():
    parser = create_parser()
    parser.description = "Select a random subset of files, then split into train, validation, and test sets."

    parser.add_argument('--num_samples', type=int, help='Total number of files to randomly select from the dataset before splitting.')

    parser.add_argument('--train_ratio', type=float, default=0.7, help='Specifies the training data ratio.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Specifies the validation data ratio.')

    parser.add_argument('--datafolder_in', type=str, required=True, help='Path to the folder containing the .npy files.')
    parser.add_argument('--datafolder_out', type=str, required=True, help='Path to save the split file names.')

    return parser

def main():
    args = create_local_parser().parse_args()

    splits = train_val_test_split_files(args.datafolder_in, args.train_ratio, args.val_ratio, num_samples=args.num_samples)

    # Save each split into its own file
    for split_name, files in splits.items():
        with open(os.path.join(args.datafolder_out, f"{split_name}_files.json"), 'w') as f:
            json.dump(files, f)

    print(f"File splits saved to {args.datafolder_out}")

if __name__ == '__main__':
    main()