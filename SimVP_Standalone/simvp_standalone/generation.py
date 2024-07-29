import os

import numpy as np
from tqdm import tqdm

from .experiment_recorder import generate_unique_id
from .preparation import normalize_data_min_max
from .simvp_utils import get_simulation_class
from .array_type import ArrayType


def update_array(array, mask, array_type, vmin=0.0, vmax=1.0, thickness=1, chance=0.2,
                 static_cells_random=False, dynamic_cells_random=False):
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


def create_array(rows, cols, array_type, vmin=0.0, vmax=1.0, thickness=1, chance=0.2,
                 static_cells_random=False, dynamic_cells_random=False):
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


def create_initials(rows, cols, num_initials, simulation_class, array_type, datafolder_out, thickness=1, chance=0.2,
                    static_cells_random=False, dynamic_cells_random=False, verbose=True):
    """
    Generates a series of samples with initial conditions and applies a stencil over iterations.

    Arguments:
    - rows: Number of rows in the array.
    - cols: Number of columns in the array.
    - num_initials: Number of initials to generate.
    - simulation_class: The Simulation class used to generate the initials.
    - array_type: The ArrayType for the initial condition.
    - datafolder_out: The path to save the initials.
    - thickness: Thickness for the lines in OUTLINE, CENTER and PLUS array types.
    - chance: Probability used for RANDOM and RANDOM_UNIFORM array types.
    - static_cells_random: For RANDOM array type. When True it fills the array with random values instead of the vmax where the mask is 1.
    - dynamic_cells_random: For RANDOM array type. When True it fills the array with random values instead of the vmin where the mask is 1.
    - verbose: If True, prints the progress of the generation.

    Returns: None
    """

    progress_iterator = range(num_initials)
    if verbose:
        progress_iterator = tqdm(progress_iterator, desc="Generating initials")

    for i in progress_iterator:
        arr, mask = create_array(rows, cols, array_type, simulation_class.vmin, simulation_class.vmax, thickness,
                                 chance, static_cells_random, dynamic_cells_random)

        unique_id = generate_unique_id(arr.tolist())
        name = f'{unique_id}_{simulation_class.__name__.lower()}_{i}'
        folder = f'{datafolder_out}/{name}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(f'{folder}/0.npy', arr)


def create_samples(total_frames, datafolder, normalize, args, verbose=True):
    """
    Generates a series of samples with initial conditions and applies a stencil over iterations.

    Arguments:
    - total_frames: Number of frames for each sample.
    - datafolder: The path to save the initials.
    - normalize: If True, normalizes the data.
    - args: Arguments for the simulation class.
    - verbose: If True, prints the progress of the generation.

    Returns: None
    """

    folders = [f for f in os.listdir(datafolder) if os.path.isdir(os.path.join(datafolder, f))]

    sims = {}
    progress_iterator = folders
    if verbose:
        progress_iterator = tqdm(progress_iterator, desc="Generating samples")

    for unique_id in progress_iterator:
        files = [f for f in os.listdir(f'{datafolder}/{unique_id}') if f.endswith('.npy')]
        if len(files) != 1:
            continue

        initial = files[0]
        simulation_name = unique_id.split('_')[1]
        if simulation_name not in sims:
            sims[simulation_name] = get_simulation_class(simulation_name)(args)

        sim = sims[simulation_name]
        try:
            arr = np.load(f'{datafolder}/{unique_id}/{initial}')
        except FileNotFoundError:
            print(f"Initial condition for {unique_id} not found, stopping generation.")
            return

        _, samples = sim.apply(arr, arr, total_frames - 1, save_history=True)
        for j, sample in enumerate(samples):
            if normalize:
                sample = normalize_data_min_max(sample, sim.vmin, sim.vmax)
            elif j == 0:
                continue
            np.save(f'{datafolder}/{unique_id}/{j}.npy', sample)
