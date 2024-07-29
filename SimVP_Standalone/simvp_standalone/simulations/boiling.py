import numpy as np

from .simulation import Simulation
from ..array_type import ArrayType

default_args = {
    'increment': 5
}


class Boiling(Simulation):
    vmin, vmax, cmap, diff_cmap, array_type = 0.0, 212.0, 'Purples', 'Oranges', ArrayType.RANDOM_UNIFORM

    def __init__(self, args=None):
        super().__init__()

        if args is None:
            args = {}

        # Update args with defaults if not present
        for key, value in default_args.items():
            args.setdefault(key, value)

        self.increment = args['increment']

    def apply(self, samples, mask, iterations, save_history=True):
        """
        Simulates boiling on a series of samples.

        Arguments:
        - samples: Numpy array of samples to apply the simulation to.
        - mask: Numpy array mask to determine which cells in the samples will be updated.
        - iterations: Number of iterations to apply the simulation.
        - save_history: When True, each iteration's result is saved and returned in an array.

        Returns:
        - The final array of samples after applying the simulation.
        - When save_history is True, also returns an array with the history of the iterations.
        """
        history = np.copy(samples)
        for _ in range(iterations):
            diffusion_to_each_neighbor = samples * (1 / 8)
            samples = np.zeros_like(samples)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    samples += np.roll(np.roll(diffusion_to_each_neighbor, dx, axis=0), dy, axis=1)

            samples = ((samples + self.increment) % self.vmax) + self.vmin

            if save_history:
                history = np.append(history, np.copy(samples))

        if save_history:
            history = np.reshape(history, (-1, samples.shape[0], samples.shape[1]))

        if save_history:
            history = np.array(history)
            return samples, history
        else:
            return samples, None
