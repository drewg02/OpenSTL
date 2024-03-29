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


def create_initials(rows, cols, num_initials, simulation, array_type, datafolder_out, thickness=1, chance=0.2,
                   static_cells_random=False, dynamic_cells_random=False, verbose=True):
    """
    Generates a series of samples with initial conditions and applies a stencil over iterations.

    Arguments:
    - rows: Number of rows in the array.
    - cols: Number of columns in the array.
    - num_initials: Number of initials to generate.
    - simulation: The Simulation class used to generate the initials.
    - array_type: The ArrayType for the initial condition.
    - save_path: The path to save the initials.
    - thickness: Thickness for the lines in OUTLINE, CENTER and PLUS array types.
    - chance: Probability used for RANDOM and RANDOM_UNIFORM array types.
    - static_cells_random: For RANDOM array type. When True it fills the array with random values instead of the vmax where the mask is 1.
    - dynamic_cells_random: For RANDOM array type. When True it fills the array with random values instead of the vmin where the mask is 1.
    - verbose: If True, prints the progress of the generation.

    Returns: None
    """

    start_time = time.time()
    last_progress = 0
    for i in range(num_initials):
        arr, mask = create_array(rows, cols, array_type, simulation.vmin, simulation.vmax, thickness, chance,
                                 static_cells_random, dynamic_cells_random)

        np.save(f'{datafolder_out}/initial_{i}.npy', arr)

        progress = ((i + 1) / num_initials) * 100
        if verbose and progress - last_progress >= 1:
            last_progress = progress
            elapsed_time = time.time() - start_time
            print(f"{progress:.2f}% done, generated {i + 1}/{num_initials} initials, {elapsed_time:.2f} seconds elapsed")


def create_samples(num_samples, total_frames, simulation, datafolder_in, datafolder_out, verbose=True):
    """
    Generates a series of samples with initial conditions and applies a stencil over iterations.

    Arguments:
    - num_samples: Number of samples to generate.
    - total_frames: Number of frames for each sample.
    - simulation: The Simulation class used to apply the simulation.
    - save_path: The path to save the initials.
    - verbose: If True, prints the progress of the generation.

    Returns: Nones
    """
    start_time = time.time()
    last_progress = 0
    for i in range(num_samples):
        try:
            arr = np.load(f'{datafolder_in}/initial_{i}.npy')
        except FileNotFoundError:
            print(f"Initial condition {i} not found, stopping generation.")
            return

        _, samples = simulation.apply(arr, arr, total_frames - 1, save_history=True)

        np.save(f'{datafolder_out}/sample_{i}.npy', samples)

        progress = ((i + 1) / num_samples) * 100
        if verbose and progress - last_progress >= 1:
            last_progress = progress
            elapsed_time = time.time() - start_time
            print(f"{progress:.2f}% done, generated {i + 1}/{num_samples} samples, {elapsed_time:.2f} seconds elapsed")
