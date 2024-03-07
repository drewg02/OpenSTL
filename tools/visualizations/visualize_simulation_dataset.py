import argparse
import time
from openstl.simulation.simulations import simulations
from openstl.simulation.visualization import save_dataset_visualizations

def main():
    parser = argparse.ArgumentParser(description="Create visualizations for a dataset.")
    parser.add_argument("dataset", type=str, help="Dataset path")
    parser.add_argument("save_path", type=str, help="Folder path to save the visualizations")
    parser.add_argument("--simulation", type=str, choices=[simulation.__name__ for simulation in simulations],
                        help="Select the type of simulation to use.", required=True)

    args = parser.parse_args()
    dataset = args.dataset
    save_path = args.save_path

    simulation_class = [simulation for simulation in simulations if simulation.__name__ == args.simulation]
    if len(simulation_class) == 0:
        raise ValueError(f'Invalid simulation class {args.simulation}')

    simulation_class = simulation_class[0]

    print(f'Starting visualization of dataset {dataset}')
    start_time = time.time()

    save_dataset_visualizations(dataset, 10, 10, simulation_class, save_path=save_path)

    print(f'Finished visualization of dataset {dataset}, took {time.time() - start_time} seconds')

if __name__ == "__main__":
    main()
