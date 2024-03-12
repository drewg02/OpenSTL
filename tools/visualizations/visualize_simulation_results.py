import argparse
import time
from openstl.simulation.simulations import simulations
from openstl.simulation.visualization import save_result_visualizations

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create visualizations for each experiment.")
    parser.add_argument("--simulation", type=str, required=True,
                        choices=[simulation.__name__ for simulation in simulations],
                        help="Select the type of simulation to use.")
    parser.add_argument('--res_dir', type=str, default='work_dirs', help="Directory to save results.")
    parser.add_argument("ex_names", nargs='+', type=str, help="Experiment names")

    args = parser.parse_args()

    # Find the simulation class based on the given argument
    simulation_class = next((simulation for simulation in simulations if simulation.__name__ == args.simulation), None)
    if simulation_class is None:
        raise ValueError(f'Invalid simulation class {args.simulation}')

    # Visualize the results for each experiment
    for ex_name in args.ex_names:
        print(f'Starting visualization of experiment {ex_name}')
        start_time = time.time()

        save_result_visualizations(args.res_dir, ex_name, simulation_class)

        elapsed_time = time.time() - start_time
        print(f'Finished visualization of experiment {ex_name}, took {elapsed_time:.2f} seconds')

if __name__ == "__main__":
    main()
