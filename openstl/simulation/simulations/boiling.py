import numpy as np

from .. import ArrayType
from .simulation import Simulation

default_args = {
    'increment': 5
}

class Boiling(Simulation):
    vmin, vmax, cmap, diff_cmap, array_type = 0.0, 212.0, 'Purples', 'Oranges', ArrayType.RANDOM_UNIFORM

    def __init__(self, args=None):
        super().__init__()
        if not args:
            args = default_args

        for key, value in default_args.items():
            if not hasattr(args, key):
                setattr(args, key, value)

        self.args = args
        self.increment = self.args.increment

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

        return (samples, np.array(history)) if save_history else (samples, None)