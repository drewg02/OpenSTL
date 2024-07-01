import os
import json
import argparse
import hashlib
import time
from tqdm import tqdm

CACHE_DIR = '.search'
CACHE_FILE = 'cache.json'

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_cache(cache_dir):
    cache_path = os.path.join(cache_dir, CACHE_FILE)
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, CACHE_FILE)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=4)

def find_files_with_param(res_dir, search_param, search_value, condition, negate):
    cache_dir = os.path.join(res_dir, CACHE_DIR)
    cache = load_cache(cache_dir)
    matching_files = []

    # Collect all model_param.json file paths
    model_files = []
    for subdir, _, files in os.walk(res_dir):
        for file in files:
            if file == 'model_param.json':
                model_files.append(os.path.join(subdir, file))

    # Create a progress bar and iterate over collected files
    for file_path in tqdm(model_files, desc="Searching"):
        file_hash = get_file_hash(file_path)

        if file_path in cache and cache[file_path]['hash'] == file_hash:
            data = cache[file_path]['data']
        else:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                cache[file_path] = {
                    'hash': file_hash,
                    'data': data,
                    'timestamp': time.time()
                }
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {file_path}: {e}")
                continue

        if search_param in data:
            param_value = str(data[search_param])
            search_value = str(search_value)
            match = False

            if condition == 'equals':
                match = param_value == search_value
            elif condition == 'contains':
                match = search_value in param_value
            elif condition == 'greater_than':
                match = float(param_value) > float(search_value)
            elif condition == 'less_than':
                match = float(param_value) < float(search_value)

            if negate:
                match = not match

            if match:
                matching_files.append(os.path.dirname(file_path))

    save_cache(cache, cache_dir)
    return matching_files

def main():
    parser = argparse.ArgumentParser(description='Search for specific parameters in model_param.json files.')
    parser.add_argument('--res_dir', type=str, default='work_dirs', help='Root directory to search in (default: work_dirs)')
    parser.add_argument('search_param', type=str, help='Parameter key to search for')
    parser.add_argument('search_value', type=str, help='Value the parameter should have')
    parser.add_argument('condition', type=str, choices=['equals', 'contains', 'greater_than', 'less_than'], help='Condition for matching the parameter value')
    parser.add_argument('--negate', action='store_true', help='Negate the condition')

    args = parser.parse_args()

    matching_directories = find_files_with_param(args.res_dir, args.search_param, args.search_value, args.condition, args.negate)

    print("Directories with matching parameter:")
    for directory in matching_directories:
        print(directory)

if __name__ == '__main__':
    main()
