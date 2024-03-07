import argparse
import time
from openstl.simulation.simulations import simulations
from openstl.simulation.visualization import save_result_visualizations

def main():
    parser = argparse.ArgumentParser(description="Create visualizations for each experiment.")
    parser.add_argument("ex_names", nargs='+', type=str, help="Ex names")
    parser.add_argument("--simulation", type=str, choices=[simulation.__name__ for simulation in simulations],
                        help="Select the type of simulation to use.", required=True)

    args = parser.parse_args()
    ex_names = args.ex_names

    simulation_class = [simulation for simulation in simulations if simulation.__name__ == args.simulation]
    if len(simulation_class) == 0:
        raise ValueError(f'Invalid simulation class {args.simulation}')

    simulation_class = simulation_class[0]

    for ex_name in ex_names:
        print(f'Starting visualization of ex {ex_name}')
        start_time = time.time()

        save_result_visualizations(ex_name, 10, 10, simulation_class)

        print(f'Finished visualization of ex {ex_name}, took {time.time() - start_time} seconds')

if __name__ == "__main__":
    main()
