class Simulation:
    vmin, vmax, cmap, diff_cmap, array_type = None, None, None, None, None

    def __init__(self, args=None):
        self.args = args

    def apply(self, arr, mask, iterations, save_history=True):
        """
        Simulates on a series of samples.

        Arguments:
        - samples: Numpy array of samples to apply the simulation to.
        - mask: Numpy array mask to determine which cells in the samples will be updated.
        - iterations: Number of iterations to apply the simulation.
        - save_history: When True, each iteration's result is saved and returned in an array.

        Returns:
        - The final array of samples after applying the simulation.
        - When save_history is True, also returns an array with the history of the iterations.
        """

        return None, None
