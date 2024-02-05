import os
import time
from enum import Enum

import numpy as np
import pickle

from openstl.utils import create_parser


class ArrayType(Enum):
    OUTLINE = 'outline'
    CENTER = 'center'
    PLUS = 'plus'
    RANDOM = 'random'
    RANDOM_UNIFORM = 'random_uniform'


def update_array(array, mask, rows, cols, array_type, thickness=1, chance=0.2, static_cells_random=False,
                 dynamic_cells_random=False):
    match array_type:
        case ArrayType.OUTLINE:
            for x in [array, mask]:
                x[0:thickness, :] = 1.0
                x[-thickness:, :] = 1.0
                x[:, 0:thickness] = 1.0
                x[:, -thickness:] = 1.0
        case ArrayType.CENTER:
            for x in [array, mask]:
                x[rows // 2 - thickness:rows // 2 + thickness, cols // 2 - thickness:cols // 2 + thickness] = 1.0
        case ArrayType.PLUS:
            for x in [array, mask]:
                x[rows // 2 - thickness:rows // 2 + thickness, :] = 1.0
                x[:, cols // 2 - thickness:cols // 2 + thickness] = 1.0
        case ArrayType.RANDOM:
            mask[:] = np.random.choice([0.0, 1.0], size=(rows, cols), p=[1.0 - chance, chance])

            random_values = np.random.uniform(low=0.0, high=1.0, size=(rows, cols))

            array[mask == 1] = np.where(static_cells_random, random_values[mask == 1], 1.0)
            array[mask == 0] = np.where(dynamic_cells_random, random_values[mask == 0], 0.0)
        case ArrayType.RANDOM_UNIFORM:
            for x in [array, mask]:
                x[:] = np.random.uniform(low=0.0, high=1.0, size=(rows, cols))
        case _:
            raise ValueError(f"Invalid array type: {array_type}")

    return array, mask


def create_array(rows, cols, array_type, thickness=1, chance=0.2, static_cells_random=False, dynamic_cells_random=False,
                 input_array=None, input_mask=None):
    arr = np.zeros((rows, cols), dtype=np.float32) if input_array is None else input_array
    mask = np.zeros((rows, cols), dtype=np.float32) if input_mask is None else input_mask

    try:
        arr, mask = update_array(arr, mask, rows, cols, array_type, thickness=thickness, chance=chance,
                                 static_cells_random=static_cells_random, dynamic_cells_random=dynamic_cells_random)
    except ValueError as e:
        print(f"Couldn't create the array and mask: {e}")
        return None, None

    return arr, mask


def apply_stencil(arr, mask, iterations, save_history=True, verbose=True):
    start_time = time.time()

    plate_history = np.copy(arr)
    new_array = np.copy(arr)

    last_progress = 0
    for f in range(iterations):
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if mask[i][j] == 1:
                    continue

                total = 0
                count = 0

                for x in range(i - 1, i + 2):
                    for y in range(j - 1, j + 2):
                        if 0 <= x < len(arr) and 0 <= y < len(arr[i]):
                            total += arr[x][y]
                            count += 1

                new_array[i][j] = total / count

        arr, new_array = new_array, arr
        if save_history:
            plate_history = np.append(plate_history, np.copy(arr))

        progress = ((f + 1) / iterations) * 100
        if progress - last_progress >= 1:
            last_progress = progress
            elapsed_time = time.time() - start_time
            if verbose: print(
                f"{progress:.2f}% done, iteration {f + 1}/{iterations}, {elapsed_time:.2f} seconds elapsed")

    if save_history:
        plate_history = np.reshape(plate_history, (-1, arr.shape[0], arr.shape[1]))

    return arr, plate_history


def create_plates(rows, cols, num_initials, total_frames, array_type=ArrayType.RANDOM, thickness=1, chance=0.2,
                  static_cells_random=False, dynamic_cells_random=False, verbose=True):
    plates = np.empty((num_initials, total_frames, rows, cols), dtype=np.float32)
    last_progress = 0

    start_time = time.time()
    for i in range(num_initials):
        # Generate the input and mask arrays
        input_array, mask_array = create_array(rows, cols, array_type, thickness=thickness, chance=chance,
                                               static_cells_random=static_cells_random,
                                               dynamic_cells_random=dynamic_cells_random)

        # Run the stencil filter on the input array for 1 iteration
        _, plate_history = apply_stencil(input_array, mask_array, total_frames - 1, verbose=False)

        # Add the plate history to the plates array
        plates[i] = plate_history

        progress = ((i + 1) / num_initials) * 100
        if progress - last_progress >= 1:
            last_progress = progress
            elapsed_time = time.time() - start_time
            if verbose: print(
                f"{progress:.2f}% done, generated {i + 1}/{num_initials} plates, {elapsed_time:.2f} seconds elapsed")

    return plates


# Split the data into train, val, and test sets
# def split_data(data, train_ratio=0.7, val_ratio=0.15):
#    train_size = int(len(data) * train_ratio)
#    val_size = int(len(data) * val_ratio)
#    test_size = len(data) - train_size - val_size
#
#    train_data = data[:train_size]
#    val_data = data[train_size:train_size + val_size]
#    test_data = data[-test_size:]
#
#    return {
#        'train': train_data,
#        'val': val_data,
#        'test': test_data
#    }

def split_data(data, train_ratio=0.7, val_ratio=0.15):
    shuffled_indices = np.random.permutation(len(data))

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]

    train_data = data[train_indices]
    val_data = data[val_indices]
    test_data = data[test_indices]

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }


def save_split(plates, train_ratio, val_ratio, pre_seq_length, file_path):
    split_pairs = split_data(plates, train_ratio=train_ratio, val_ratio=val_ratio)

    dataset = {}
    for split in ['train', 'val', 'test']:
        data_x = split_pairs[split][:, :pre_seq_length, ...]
        data_y = split_pairs[split][:, pre_seq_length:, ...]

        dataset['X_' + split], dataset['Y_' + split] = data_x, data_y

    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # save as a pkl file
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)


def main():
    parser = create_parser()

    parser.add_argument('--num_plates', type=int, default=50)

    parser.add_argument('--image_height', type=int, default=64)
    parser.add_argument('--image_width', type=int, default=64)
    parser.add_argument('--array_type', type=ArrayType, default=ArrayType.RANDOM)
    parser.add_argument('--chance', type=float, default=0.1)
    parser.add_argument('--static_cells_random', type=bool, default=False)
    parser.add_argument('--dynamic_cells_random', type=bool, default=False)

    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)

    parser.add_argument('--datafile', type=str)
    parser.add_argument('--overwrite_datafile', type=bool, default=False)

    parser.add_argument('--multi_num_plates', nargs='+', type=int)

    args = parser.parse_args()

    file_path = args.datafile
    if not args.overwrite_datafile and os.path.exists(file_path):
        print(f"File {file_path} already exists. Set --overwrite_datafile' to True to overwrite.")
        return

    num_plates = args.num_plates

    pre_seq_length = args.pre_seq_length if args.pre_seq_length is not None else 10
    aft_seq_length = args.aft_seq_length if args.aft_seq_length is not None else 10
    total_seq_length = pre_seq_length + aft_seq_length

    image_height = args.image_height
    image_width = args.image_width
    array_type = args.array_type
    chance = args.chance
    static_cells_random = args.static_cells_random
    dynamic_cells_random = args.dynamic_cells_random

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio

    # Check that the ratios are valid
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be less than 1.0")

    start_time = time.time()
    plates = create_plates(image_height, image_width, num_plates, total_seq_length, chance=chance,
                           array_type=array_type, static_cells_random=static_cells_random,
                           dynamic_cells_random=dynamic_cells_random, verbose=True)

    # Reshape the plates array to be (num_plates, total_seq_length, channels, rows, cols)
    plates = np.reshape(plates, (num_plates, total_seq_length, 1, image_height, image_width))

    elapsed = time.time() - start_time
    print(f"Generating {len(plates)} plates took {elapsed} seconds.")

    save_split(plates, train_ratio, val_ratio, pre_seq_length, file_path)

    if args.multi_num_plates is not None:
        for supplemental_num_plates in args.multi_num_plates:
            supplemental_plates = plates[supplemental_num_plates:]
            supplemental_file_path = file_path.replace(f'_{num_plates}plates', f'_{supplemental_num_plates}plates')
            save_split(supplemental_plates, train_ratio, val_ratio, pre_seq_length, supplemental_file_path)


if __name__ == '__main__':
    main()
