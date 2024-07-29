import numpy as np
from scipy.ndimage import convolve

from .simulation import Simulation
from ..array_type import ArrayType


class HeatTransfer(Simulation):
    vmin, vmax, cmap, diff_cmap, array_type = 0.0, 1.0, 'coolwarm', 'Greens', ArrayType.RANDOM

    def __init__(self, args=None):
        super().__init__(args)
        self.args = args

    #
    # Original implementation, new should provide the exact same results but be faster.
    #
    # def apply(self, samples, mask, iterations, save_history=True):
    #     """
    #     Simulates heat transfer on a series of samples.
    #
    #     Arguments:
    #     - samples: Numpy array of samples to apply the simulation to.
    #     - mask: Numpy array mask to determine which cells in the samples will be updated.
    #     - iterations: Number of iterations to apply the simulation.
    #     - save_history: When True, each iteration's result is saved and returned in an array.
    #
    #     Returns:
    #     - The final array of samples after applying the simulation.
    #     - When save_history is True, also returns an array with the history of the iterations.
    #     """
    #     samples = np.copy(samples)
    #
    #     history = np.copy(samples)
    #     new_samples = np.copy(samples)
    #     for _ in range(iterations):
    #         for i in range(len(samples)):
    #             for j in range(len(samples[i])):
    #                 if mask[i][j] == 1:
    #                     continue
    #
    #                 total = 0
    #                 count = 0
    #
    #                 for x in range(i - 1, i + 2):
    #                     for y in range(j - 1, j + 2):
    #                         if 0 <= x < len(samples) and 0 <= y < len(samples[i]):
    #                             total += samples[x][y]
    #                             count += 1
    #
    #                 new_samples[i][j] = total / count
    #
    #         samples, new_samples = new_samples, samples
    #         if save_history:
    #             history = np.append(history, np.copy(samples))
    #
    #     if save_history:
    #         history = np.reshape(history, (-1, samples.shape[0], samples.shape[1]))
    #
    #     return (samples, np.array(history)) if save_history else (samples, None)

    def apply(self, samples, mask, iterations, save_history=True):
        """
        Simulates heat transfer on a series of samples.

        Arguments:
        - samples: Numpy array of samples to apply the simulation to.
        - mask: Numpy array mask to determine which cells in the samples will be updated.
        - iterations: Number of iterations to apply the simulation.
        - save_history: When True, each iteration's result is saved and returned in an array.

        Returns:
        - The final array of samples after applying the simulation.
        - When save_history is True, also returns an array with the history of the iterations.
        """
        samples = np.copy(samples)
        mask = np.copy(mask)

        kernel = np.ones((3, 3))
        history = [samples.copy()] if save_history else None

        for _ in range(iterations):
            convolved = convolve(samples, kernel, mode='constant', cval=0.0)
            counts = convolve(np.ones_like(samples), kernel, mode='constant', cval=0.0)
            new_samples = np.where(mask, samples, convolved / counts)
            samples[:] = new_samples

            if save_history:
                history.append(samples.copy())

        if save_history:
            history = np.array(history)
            return samples, history
        else:
            return samples, None