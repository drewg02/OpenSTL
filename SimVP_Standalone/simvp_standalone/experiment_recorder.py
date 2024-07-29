import hashlib
import json
import os


def generate_experiment_record(**params):
    """Generate a dictionary for the experiment with a unique ID."""
    # Serialize the dictionary (excluding 'id' if present) to generate a unique ID
    unique_id = generate_unique_id({key: params[key] for key in params if key != 'id'})
    params['id'] = unique_id
    print(f'Unique Experiment ID: {unique_id}')
    return params


def save_experiment_record(experiment_record, filename):
    """Save the experiment record to a JSON file, creating directory if needed."""
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist. Creating.")
        os.makedirs(directory)
    with open(filename, 'w') as file:
        json.dump(experiment_record, file, indent=4)


def load_experiment_record(filename):
    """Load the experiment record from a JSON file."""
    with open(filename, 'r') as file:
        return json.load(file)


def generate_unique_id(experiment_record):
    """Generate a SHA-256 hash as a unique ID for the experiment record."""
    serialized_record = json.dumps(experiment_record, sort_keys=True)
    hash_object = hashlib.sha256(serialized_record.encode())
    return hash_object.hexdigest()
