import os
import time
from enum import Enum
import numpy as np
import pickle
from openstl.utils import create_parser
from openstl.simulations import HeatTransfer, Boiling


class ArrayType(Enum):
    OUTLINE = 'outline'
    CENTER = 'center'
    PLUS = 'plus'
    RANDOM = 'random'
    RANDOM_UNIFORM = 'random_uniform'

def update_array(array, mask, array_type, vmin=0.0, vmax=1.0, thickness=1, chance=0.2, static_cells_random=False, dynamic_cells_random=False):
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
    rows, cols = array.shape
    if array_type == ArrayType.OUTLINE:
        array[:thickness, :] = array[-thickness:, :] = array[:, :thickness] = array[:, -thickness:] = vmax
        mask[:thickness, :] = mask[-thickness:, :] = mask[:, :thickness] = mask[:, -thickness:] = 1
    
    elif array_type == ArrayType.CENTER:
        mid_row, mid_col = rows // 2, cols // 2
        center_slice = (slice(mid_row-thickness, mid_row+thickness), slice(mid_col-thickness, mid_col+thickness))
        array[center_slice] = vmax
        mask[center_slice] = 1

    elif array_type == ArrayType.PLUS:
        array[rows // 2 - thickness:rows // 2 + thickness, :] = vmax
        array[:, cols // 2 - thickness:cols // 2 + thickness] = vmax
        mask[rows // 2 - thickness:rows // 2 + thickness, :] = 1
        mask[:, cols // 2 - thickness:cols // 2 + thickness] = 1

    elif array_type == ArrayType.RANDOM:
        mask[:] = np.random.choice([0, 1], size=(rows, cols), p=[1-chance, chance])
        random_values = np.random.uniform(vmin, vmax, size=(rows, cols))
        array[mask == 1] = random_values[mask == 1] if static_cells_random else vmax
        array[mask == 0] = random_values[mask == 0] if dynamic_cells_random else vmin

    elif array_type == ArrayType.RANDOM_UNIFORM:
        array[:] = np.random.uniform(vmin, vmax, size=(rows, cols))
        mask[:] = np.random.choice([0, 1], size=(rows, cols), p=[1-chance, chance])

    else:
        raise ValueError(f"Invalid array type: {array_type}")

    return array, mask

def create_array(rows, cols, array_type, vmin=0.0, vmax=1.0, thickness=1, chance=0.2, static_cells_random=False, dynamic_cells_random=False):
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
    arr, mask = update_array(arr, mask, array_type, vmin, vmax, thickness, chance, static_cells_random, dynamic_cells_random)
    return arr, mask

def create_samples(rows, cols, num_samples, total_frames, simulation, array_type, thickness=1, chance=0.2, static_cells_random=False, dynamic_cells_random=False, verbose=True):
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
        arr, mask = create_array(rows, cols, array_type, simulation.vmin, simulation.vmax, thickness, chance, static_cells_random, dynamic_cells_random)
        _, samples[i] = simulation.apply(arr, mask, total_frames - 1, save_history=True)

        progress = ((i + 1) / num_samples) * 100
        if verbose and progress - last_progress >= 1:
            last_progress = progress
            elapsed_time = time.time() - start_time
            print(f"{progress:.2f}% done, generated {i + 1}/{num_samples} samples, {elapsed_time:.2f} seconds elapsed")
    
    return samples

def normalize_data_min_max(data, vmin, vmax):
    """
    Normalizes the dataset using min-max scaling to a range of [0, 1].

    Arguments:
    - data: The dataset to normalize.
    - min_val: The minimum value of the input data.
    - max_val: The maximum value of the input data.

    Returns:
    - The normalized data scaled to range [0, 1].
    """

    return (data - vmin) / (vmax - vmin)

def split_data(data, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the data into training, validation, and test sets.

    Arguments:
    - data: The dataset to split.
    - train_ratio: Proportion of the dataset to include in the training set.
    - val_ratio: Proportion of the dataset to include in the validation set.

    Returns:
    - A dictionary with 'train', 'val', and 'test' keys, each of which contains their respective split of the data.
    """
    np.random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    test_size = len(data) - train_size - val_size

    return {
        'train': data[:train_size],
        'val': data[train_size:train_size + val_size],
        'test': data[-test_size:]
    }

def save_split(split_pairs, pre_seq_length, file_path):
    """
    Saves the split data into a numpy pickle file in input and output pairs (X and Y respectively).

    Arguments:
    - split_pairs: A dictionary with 'train', 'val', and 'test' keys, each of which contains their respective split of the data.
    - pre_seq_length: The length of the sequence that is the input for the model.
    - file_path: The path to the file where the data will be saved.
    """
    dataset = {}
    for split in ['train', 'val', 'test']:
        dataset['X_' + split] = split_pairs[split][:, :pre_seq_length, ...]
        dataset['Y_' + split] = split_pairs[split][:, pre_seq_length:, ...]

    folder_path = os.path.dirname(file_path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)


def create_local_parser():
    parser = create_parser()

    parser.add_argument('--simulation', type=str, choices=[HeatTransfer.__name__, Boiling.__name__],
                    help='Select the type of simulation to run.', required=True)
    parser.add_argument('--num_samples', nargs='+', type=int, required=True)

    parser.add_argument('--image_height', type=int, default=64)
    parser.add_argument('--image_width', type=int, default=64)
    parser.add_argument('--array_type', type=ArrayType, default=ArrayType.RANDOM)
    parser.add_argument('--chance', type=float, default=0.1)
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--static_cells_random', type=bool, default=False)
    parser.add_argument('--dynamic_cells_random', type=bool, default=False)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--increment', type=int, default=5)
    parser.add_argument('--normalize', type=bool, default=True)

    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)

    parser.add_argument('--datafile', type=str, required=True)
    parser.add_argument('--overwrite_datafile', type=bool, default=False)

    return parser

def main():
    parser = create_local_parser()
    args = parser.parse_args()

    file_path = args.datafile
    if not args.overwrite_datafile and os.path.exists(file_path):
        print(f"File {file_path} already exists. Set --overwrite_datafile' to True to overwrite.")
        return

    num_samples = max(args.num_samples)

    match args.simulation:
        case HeatTransfer.__name__:
            simulation = HeatTransfer()
        case Boiling.__name__:
            simulation = Boiling(args.offset, args.increment)
        case _:
            raise ValueError(f"Invalid simulation type: {args.simulation}")

    pre_seq_length = args.pre_seq_length if args.pre_seq_length is not None else 10
    aft_seq_length = args.aft_seq_length if args.aft_seq_length is not None else 10
    total_seq_length = pre_seq_length + aft_seq_length

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({args.train_ratio}) + val_ratio ({args.val_ratio}) must be less than 1.0")

    start_time = time.time()
    samples = create_samples(args.image_height, args.image_width, num_samples, total_seq_length, simulation, args.array_type, args.thickness, args.chance, args.static_cells_random, args.dynamic_cells_random, verbose=True)
    samples = np.reshape(samples, (num_samples, total_seq_length, 1, args.image_height, args.image_width))

    elapsed = time.time() - start_time
    print(f"Generating {len(samples)} samples took {elapsed} seconds.")

    if args.normalize:
        samples = normalize_data_min_max(samples, simulation.vmin, simulation.vmax)
    split_pairs = split_data(samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    save_split(split_pairs, pre_seq_length, file_path)

    if len(args.num_samples) > 1:
        for num_samples_additional in args.num_samples:
            train_amount = int(num_samples_additional * args.train_ratio)
            val_amount = int(num_samples_additional * args.val_ratio)
            test_amount = num_samples_additional - train_amount - val_amount
            supplemental_split_pairs =  {
                'train': split_pairs['train'][:train_amount],
                'val': split_pairs['val'][:val_amount],
                'test': split_pairs['test'][:test_amount]
            }

            supplemental_file_path = file_path.replace(f'_{num_samples}samples', f'_{num_samples_additional}samples')
            save_split(supplemental_split_pairs, pre_seq_length, supplemental_file_path)
    

if __name__ == '__main__':
    main()
