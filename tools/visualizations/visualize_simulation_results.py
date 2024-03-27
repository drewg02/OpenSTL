import argparse
import time
from openstl.simulation.simulations import simulations
from openstl.simulation.visualization import load_results, save_result_images

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create visualizations for each experiment.")
    parser.add_argument("--simulation", type=str, required=True,
                        choices=[simulation.__name__ for simulation in simulations],
                        help="Select the type of simulation to use.")
    parser.add_argument('--res_dir', type=str, default='work_dirs', help="Directory to save results.")
    parser.add_argument('--output_dir', type=str, help="Directory to save visualizations.")
    parser.add_argument("--ex_name", type=str, help="Experiment name")

    args = parser.parse_args()

    # Find the simulation class based on the given argument
    simulation_class = next((simulation for simulation in simulations if simulation.__name__ == args.simulation), None)
    if simulation_class is None:
        raise ValueError(f'Invalid simulation class {args.simulation}')

    # Visualize the results
    print(f'Starting visualization of experiment {args.ex_name}')
    start_time = time.time()

    inputs, trues, preds = load_results(args.ex_name, args.res_dir)
    save_result_images(inputs, trues, preds, simulation_class, args.output_dir)

    elapsed_time = time.time() - start_time
    print(f'Finished visualization of experiment {args.ex_name}, took {elapsed_time:.2f} seconds')

if __name__ == "__main__":
    main()
