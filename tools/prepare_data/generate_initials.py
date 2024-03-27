import os
import time

from openstl.utils import create_parser

from openstl.simulation.simulations import simulations
from openstl.simulation.generation import ArrayType, create_initials
from openstl.simulation.utils import get_simulation_class


# Create a parser with additional arguments specific to dataset generation.
def create_local_parser():
    parser = create_parser()
    parser.description = "Generate a dataset for a simulation."

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                        help='Determines the simulation type.', required=True)

    parser.add_argument('--num_initials', type=int,
                        help='Specifies the number of initials.', required=True)
    parser.add_argument('--image_height', type=int, default=64,
                        help='Specifies the image height.')
    parser.add_argument('--image_width', type=int, default=64,
                        help='Specifies the image width.')

    parser.add_argument('--array_type', type=ArrayType, default=ArrayType.RANDOM,
                        help='Defines the array type.')
    parser.add_argument('--chance', type=float, default=0.1,
                        help='Sets the chance parameter.')
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--static_cells_random', action='store_true')
    parser.add_argument('--dynamic_cells_random', action='store_true')
    parser.add_argument('--increment', type=int, default=5,
                        help="Only applies to Boiling simulation, sets the increment value.")

    parser.add_argument('--datafolder_out', type=str, required=True,
                        help='Specifies the data folder path.')

    return parser


def main():
    args = create_local_parser().parse_args()  # Parse the command line arguments.

    # Get the simulation class.
    simulation_class = get_simulation_class(args.simulation)
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {args.simulation}")

    simulation = simulation_class(args)  # Initialize the simulation class.

    start_time = time.time()  # Record the start time.
    # Generate initials based on the provided parameters.
    create_initials(args.image_height, args.image_width, args.num_initials, simulation,
                             args.array_type, args.datafolder_out, args.thickness, args.chance, args.static_cells_random,
                             args.dynamic_cells_random, verbose=True)

    elapsed = time.time() - start_time  # Calculate the elapsed time.
    print(f"Generating {args.num_initials} initials took {elapsed} seconds.")  # Print the time taken to generate initials.


if __name__ == '__main__':
    main()
