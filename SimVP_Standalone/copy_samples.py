import time
import os

from argparse import ArgumentParser as ArgParser
from simvp_standalone.experiment_recorder import generate_experiment_record, save_experiment_record

from simvp_standalone.preparation import check_samples, copy_samples


# Create a parser with additional arguments specific to dataset generation.
def create_local_parser():
    parser = ArgParser()

    parser.description = "Copies data into new directory while che."

    parser.add_argument('--datafolders', type=str, required=True, nargs='+',
                        help='Specifies the data folder paths.')
    parser.add_argument('--new_datafolder', type=str, required=True,
                        help='Specifies the new data folder path.')
    parser.add_argument('--move', action='store_true',
                        help='If set, moves the files instead of copying.')

    return parser


def main():
    args = create_local_parser().parse_args()  # Parse the command line arguments.

    start_time = time.time()  # Record the start time.
    # Generate samples based on the provided parameters.
    passed_check = check_samples(args.datafolders, verbose=True)
    if passed_check:
        if not os.path.exists(args.new_datafolder):
            print(f"Path '{args.new_datafolder}' does not exist. Creating...")
            os.makedirs(args.new_datafolder)

        copy_samples(args.datafolders, args.new_datafolder, args.move, verbose=True)
    else:
        print("Samples did not pass check.")

    elapsed = time.time() - start_time  # Calculate the elapsed time.

    copy_arguments = {
        'datafolders': args.datafolders,
        'new_datafolder': args.new_datafolder,
        'move': args.move
    }

    record = generate_experiment_record(**copy_arguments)
    save_experiment_record(record, os.path.join(args.new_datafolder, f"{record['id']}_copy.json"))

    print(f"Copying samples took {elapsed} seconds.")  # Print the time taken to check and copy samples.

if __name__ == '__main__':
    main()
