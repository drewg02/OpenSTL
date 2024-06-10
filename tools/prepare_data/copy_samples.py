import time
import os

from argparse import ArgumentParser as ArgParser

from openstl.simulation.preparation import check_samples, copy_samples


# Create a parser with additional arguments specific to dataset generation.
def create_local_parser():
    parser = ArgParser()

    parser.description = "Copies data into new directory while che."

    parser.add_argument('--datafolders', type=str, required=True, nargs='+',
                        help='Specifies the data folder paths.')
    parser.add_argument('--new_datafolder', type=str, required=True,
                        help='Specifies the new data folder path.')

    return parser


def main():
    args = create_local_parser().parse_args()  # Parse the command line arguments.
    args_dict = vars(args)

    start_time = time.time()  # Record the start time.
    # Generate samples based on the provided parameters.
    passed_check = check_samples(args.datafolders, verbose=True)
    if passed_check:
        if not os.path.exists(args.new_datafolder):
            print(f"Path '{args.new_datafolder}' does not exist. Creating...")
            os.makedirs(args.new_datafolder)

        copy_samples(args.datafolders, args.new_datafolder, verbose=True)
    else:
        print("Samples did not pass check.")

    elapsed = time.time() - start_time  # Calculate the elapsed time.

    print(f"Copying samples took {elapsed} seconds.")  # Print the time taken to check and copy samples.

if __name__ == '__main__':
    main()
