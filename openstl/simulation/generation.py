import time
import numpy as np
from .array_type import ArrayType


def update_array(array, mask, array_type, vmin=0.0, vmax=1.0, thickness=1, chance=0.2, static_cells_random=False,
                 dynamic_cells_random=False):
    """
    Updates the array and mask based on the array type.

    Arguments:
    - array: Numpy array to update (values vmin to vmax).
    - mask: Numpy array mask to update (values 0 or 1).
    - array_type: The type of initial condition to apply, defined by ArrayType.
    - vmax: The maximum value to use when filling the array.
    - vmin: The minimum value to use when filling the array.
    - thickness: Thickness for the lines in OUTLINE, CENTER and PLUS array types.
    - chance: Probability used for RANDOM and RANDOM_UNIFORM array types.
    - static_cells_random: For RANDOM array type. When True it fills the array with random values instead of the vmax where the mask is 1.
    - dynamic_cells_random: For RANDOM array type. When True it fills the array with random values instead of the vmin where the mask is 1.

    Returns:
    - The final array and mask after updating.
    """

    # if the user set the chance to None or <= 0, then each time we want to select a new
    # random chance for just this call - creating a more diverse dataset - one that isn't
    # just trained on 20% chance samples, for instance.
    if chance is None or chance <= 0:
        chance = np.random.random()
        print(f"Chance was None (or <= 0) - so this initial condition we will use randomly chosen chance of: {chance}")

    rows, cols = array.shape
    if array_type == ArrayType.OUTLINE:
        array[:thickness, :] = array[-thickness:, :] = array[:, :thickness] = array[:, -thickness:] = vmax
        mask[:thickness, :] = mask[-thickness:, :] = mask[:, :thickness] = mask[:, -thickness:] = 1

    elif array_type == ArrayType.CENTER:
        mid_row, mid_col = rows // 2, cols // 2
        center_slice = (
        slice(mid_row - thickness, mid_row + thickness), slice(mid_col - thickness, mid_col + thickness))
        array[center_slice] = vmax
        mask[center_slice] = 1

    elif array_type == ArrayType.PLUS:
        array[rows // 2 - thickness:rows // 2 + thickness, :] = vmax
        array[:, cols // 2 - thickness:cols // 2 + thickness] = vmax
        mask[rows // 2 - thickness:rows // 2 + thickness, :] = 1
        mask[:, cols // 2 - thickness:cols // 2 + thickness] = 1

    elif array_type == ArrayType.RANDOM:
        mask[:] = np.random.choice([0, 1], size=(rows, cols), p=[1 - chance, chance])
        random_values = np.random.uniform(vmin, vmax, size=(rows, cols))
        array[mask == 1] = random_values[mask == 1] if static_cells_random else vmax
        array[mask == 0] = random_values[mask == 0] if dynamic_cells_random else vmin

    elif array_type == ArrayType.RANDOM_UNIFORM:
        array[:] = np.random.uniform(vmin, vmax, size=(rows, cols))
        mask[:] = np.random.choice([0, 1], size=(rows, cols), p=[1 - chance, chance])

    else:
        raise ValueError(f"Invalid array type: {array_type}")

    return array, mask


def create_array(rows, cols, array_type, vmin=0.0, vmax=1.0, thickness=1, chance=0.2, static_cells_random=False,
                 dynamic_cells_random=False):
    """
    Creates and returns a new array and mask.

    Arguments:
    - rows: Number of rows in the array.
    - cols: Number of columns in the array.
    - array_type: The type of initial condition to apply, defined by ArrayType.
    - vmax: The maximum value to use when filling the array.
    - vmin: The minimum value to use when filling the array.
    - thickness: Thickness for the lines in OUTLINE, CENTER and PLUS array types.
    - chance: Probability used for RANDOM and RANDOM_UNIFORM array types.
    - static_cells_random: For RANDOM array type. When True it fills the array with random values instead of the vmax where the mask is 1.
    - dynamic_cells_random: For RANDOM array type. When True it fills the array with random values instead of the vmin where the mask is 1.

    Returns:
    - The newly created array and mask.
    """
    arr = np.zeros((rows, cols), dtype=np.float32)
    mask = np.zeros((rows, cols), dtype=np.float32)
    arr, mask = update_array(arr, mask, array_type, vmin, vmax, thickness, chance, static_cells_random,
                             dynamic_cells_random)
    return arr, mask


def create_samples(rows, cols, num_samples, total_frames, simulation, array_type, thickness=1, chance=0.2,
                   static_cells_random=False, dynamic_cells_random=False, verbose=True):
    """
    Generates a series of samples with initial conditions and applies a stencil over iterations.

    Arguments:
    - rows: Number of rows in the array.
    - cols: Number of columns in the array.
    - num_samples: Number of samples to generate.
    - total_frames: Number of frames for each sample.
    - array_type: The ArrayType for the initial conditions of the array.
    - vmax: The maximum value to use when filling the array.
    - vmin: The minimum value to use when filling the array.
    - thickness: Thickness for the lines in OUTLINE, CENTER and PLUS array types.
    - chance: Probability used for RANDOM and RANDOM_UNIFORM array types.
    - static_cells_random: For RANDOM array type. When True it fills the array with random values instead of the vmax where the mask is 1.
    - dynamic_cells_random: For RANDOM array type. When True it fills the array with random values instead of the vmin where the mask is 1.

    Returns:
    - A 4D numpy array representing the samples over the iterations.
    """
    start_time = time.time()
    last_progress = 0
    samples = np.empty((num_samples, total_frames, rows, cols), dtype=np.float32)
    for i in range(num_samples):
        arr, mask = create_array(rows, cols, array_type, simulation.vmin, simulation.vmax, thickness, chance,
                                 static_cells_random, dynamic_cells_random)
        _, samples[i] = simulation.apply(arr, mask, total_frames - 1, save_history=True)

        progress = ((i + 1) / num_samples) * 100
        if verbose and progress - last_progress >= 1:
            last_progress = progress
            elapsed_time = time.time() - start_time
            print(f"{progress:.2f}% done, generated {i + 1}/{num_samples} samples, {elapsed_time:.2f} seconds elapsed")

    return samples
