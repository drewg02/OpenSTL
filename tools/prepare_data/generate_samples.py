import os
import time
import numpy as np
from openstl.utils import create_parser
from openstl.simulation.simulations import simulations
from openstl.simulation.generation import ArrayType, create_samples
from openstl.simulation.utils import save_data, get_simulation_class, get_seq_lengths


def create_local_parser():
    parser = create_parser()

    parser.add_argument('--simulation', type=str, choices=[simulation.__name__ for simulation in simulations],
                    help='Select the type of simulation to run.', required=True)
    parser.add_argument('--num_samples', type=int, required=True)

    parser.add_argument('--image_height', type=int, default=64)
    parser.add_argument('--image_width', type=int, default=64)
    parser.add_argument('--array_type', type=ArrayType, default=ArrayType.RANDOM)
    parser.add_argument('--chance', type=float, default=0.1)
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--static_cells_random', type=bool, default=False)
    parser.add_argument('--dynamic_cells_random', type=bool, default=False)
    parser.add_argument('--increment', type=int, default=5)

    parser.add_argument('--datafile_out', type=str, required=True)
    parser.add_argument('--overwrite_datafile', type=bool, default=False)

    return parser

def main():
    args = create_local_parser().parse_args()

    if not args.overwrite_datafile and os.path.exists(args.datafile_out):
        print(f"File {args.datafile_out} already exists. Set --overwrite_datafile' to True to overwrite.")
        return

    simulation_class = get_simulation_class(args.simulation)
    if not simulation_class:
        raise ValueError(f"Invalid simulation: {args.simulation}")

    simulation = simulation_class(args)

    pre_seq_length, aft_seq_length, total_length = get_seq_lengths(args)
    if total_length is None:
        raise ValueError("Must specify --total_length or --pre_seq_length and --aft_seq_length")

    start_time = time.time()
    samples = create_samples(args.image_height, args.image_width, args.num_samples, total_length, simulation, args.array_type, args.thickness, args.chance, args.static_cells_random, args.dynamic_cells_random, verbose=True)
    samples = np.reshape(samples, (args.num_samples, total_length, 1, args.image_height, args.image_width))

    elapsed = time.time() - start_time
    print(f"Generating {len(samples)} samples took {elapsed} seconds.")

    save_data(samples, args.datafile_out)
    

if __name__ == '__main__':
    main()
