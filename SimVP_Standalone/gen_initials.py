import os
import time

from argparse import ArgumentParser as ArgParser
from simvp_standalone.experiment_recorder import generate_experiment_record, save_experiment_record

from simvp_standalone.simulations import simulations
from simvp_standalone.generation import ArrayType, create_initials
from simvp_standalone.simvp_utils import get_simulation_class


# Create a parser with additional arguments specific to dataset generation.
def create_local_parser():
    parser = ArgParser()

    parser.description = "Generate a dataset for a simulation."

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                        help='Determines the simulation type.', required=True)

    parser.add_argument('--num_initials', type=int,
                        help='Specifies the number of initials.', required=True)
    parser.add_argument('--image_height', type=int, default=64,
                        help='Specifies the image height.')
    parser.add_argument('--image_width', type=int, default=64,
                        help='Specifies the image width.')

    parser.add_argument('--array_type', type=ArrayType,
                        help='Defines the array type.')
    parser.add_argument('--chance', type=float, default=0.1,
                        help='Sets the chance parameter.')
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--static_cells_random', action='store_true',
                        help="Only applies to HeatTransfer simulation, sets if mask cells should be random values.")
    parser.add_argument('--dynamic_cells_random', action='store_true',
                        help="Only applies to HeatTransfer simulation, sets if non mask cells should be random values.")
    parser.add_argument('--increment', type=int, default=5,
                        help="Only applies to Boiling simulation, sets the increment value.")

    parser.add_argument('--datafolder', type=str, required=True,
                        help='Specifies the data folder path.')

    return parser


def main():
    args = create_local_parser().parse_args()  # Parse the command line arguments.

    # Get the simulation class.
    simulation_class = get_simulation_class(args.simulation)
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {args.simulation}")

    if not args.array_type:
        args.array_type = simulation_class.array_type

    if not os.path.exists(args.datafolder):
        print(f"Path '{args.datafolder}' does not exist. Creating...")
        os.makedirs(args.datafolder)

    start_time = time.time()  # Record the start time.
    # Generate initials based on the provided parameters.
    create_initials(args.image_height, args.image_width, args.num_initials, simulation_class,
                             args.array_type, args.datafolder, args.thickness, args.chance, args.static_cells_random,
                             args.dynamic_cells_random, verbose=True)

    elapsed = time.time() - start_time  # Calculate the elapsed time.

    initials_arguments = {
        'simulation': args.simulation,
        'num_initials': args.num_initials,
        'image_height': args.image_height,
        'image_width': args.image_width,
        'array_type': args.array_type,
        'chance': args.chance,
        'thickness': args.chance,
        'static_cells_random': args.chance,
        'dynamic_cells_random': args.chance,
        'increment': args.increment
    }

    record = generate_experiment_record(**initials_arguments)
    save_experiment_record(record, os.path.join(args.datafolder, f"{record['id']}_initials.json"))

    print(f"Generating {args.num_initials} initials took {elapsed} seconds.")  # Print the time taken to generate initials.


if __name__ == '__main__':
    main()
