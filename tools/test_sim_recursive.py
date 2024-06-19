# Copyright (c) CAIRI AI Lab. All rights reserved

import os
import warnings
warnings.filterwarnings('ignore')

from openstl.simulation.train import SimulationExperiment
from openstl.simulation.utils import create_parser, generate_config, create_dataloader
from openstl.utils import (default_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)
import numpy as np

try:
    import nni
    has_nni = True
except ImportError:
    has_nni = False


def main():
    args = create_parser().parse_args()

    training_config = generate_config(args)

    config = args.__dict__
    config.update(training_config)

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    assert args.config_file is not None, "Config file is required for testing"
    config = update_config(config, load_config(args.config_file),
                           exclude_keys=['method', 'batch_size', 'val_batch_size'])
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]
    if not config['inference'] and not config['test']:
        config['test'] = True

    # set multi-process settings
    setup_multi_processes(config)

    exp = SimulationExperiment(args)
    saved_path = os.path.join(exp.path, 'saved')

    data = exp.test_loader.dataset.data
    sample_length = len(data['samples'][0])

    rank, _ = get_dist_info()

    final_mse = []
    for j in range(sample_length - 1):

        if config['inference'] and not config['test']:
            print('>' * 35 + f' inferencing {args.ex_name} i{j}  ' + '<' * 35)
            mse = exp.inference()
        else:
            print('>' * 35 + f' testing {args.ex_name} i{j} ' + '<' * 35)
            mse = exp.test(save_dir=args.saved_path)

        final_mse.append(mse)

        for i in range(len(data['samples'])):
            unique_id = data['samples'][i][0].split('/')[-2]
            for result_type in ['inputs', 'trues', 'preds']:
                zero_path = os.path.join(saved_path, result_type, unique_id, '0.npy')
                save_name = 'first.npy' if j == 0 else f'{j}.npy'
                save_path = os.path.join(saved_path, result_type, unique_id, save_name)

                os.rename(zero_path, save_path)

                if result_type == 'preds':
                    data['samples'][i].pop(0)
                    data['samples'][i][0] = save_path

        exp.test_loader = create_dataloader(data, args.pre_seq_length, args.aft_seq_length, args.val_batch_size, False, False, args.dist)

    for i in range(len(data['samples'])):
        unique_id = data['samples'][i][0].split('/')[-2]
        for result_type in ['inputs', 'trues', 'preds']:
            first_path = os.path.join(saved_path, result_type, unique_id, 'first.npy')
            zero_path = os.path.join(saved_path, result_type, unique_id, '0.npy')

            os.rename(first_path, zero_path)

    final_mse = sum(final_mse) / len(final_mse)
    if rank == 0 and has_nni and final_mse is not None:
        nni.report_final_result(final_mse)

if __name__ == '__main__':
    main()