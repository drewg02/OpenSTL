# Copyright (c) CAIRI AI Lab. All rights reserved

import os
import copy
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
    saved_path = args.saved_path if args.saved_path else os.path.join(exp.path, 'saved')

    data = copy.deepcopy(exp.test_loader.dataset.data)
    sample_length = len(data['samples'][0])

    rank, _ = get_dist_info()

    final_eval = {}
    final_data = {}
    for j in range(sample_length - 1):

        if config['inference'] and not config['test']:
            print('>' * 35 + f' inferencing {args.ex_name} i{j}  ' + '<' * 35)
            eval_res, result_data = exp.inference(save_files=False)
        else:
            print('>' * 35 + f' testing {args.ex_name} i{j} ' + '<' * 35)
            eval_res, result_data = exp.test(save_files=False)

        if eval_res:
            for key, value in eval_res.items():
                if key not in final_eval:
                    final_eval[key] = []

                final_eval[key].append(value)

        for i in range(len(data['samples'])):
            for result_type in ['inputs', 'trues', 'preds']:
                res_data = result_data[result_type][i]

                if result_type not in final_data:
                    final_data[result_type] = []

                if len(final_data[result_type]) <= i:
                    final_data[result_type].append([])

                final_data[result_type][i].append(res_data)

                if result_type == 'preds':
                    data['samples'][i].pop(0)
                    data['samples'][i][0] = res_data.reshape(res_data.shape[-2], res_data.shape[-1])

        exp.test_loader = create_dataloader(data, args.pre_seq_length, args.aft_seq_length, args.val_batch_size, False, False, args.dist)

    if rank == 0:
        exp.save_results(final_data, saved_path)

    for key in final_eval.keys():
        final_eval[key] = np.mean(final_eval[key])

    if rank == 0 and has_nni and 'mse' in final_eval:
        nni.report_final_result(final_eval['mse'])

        eval_log = 'Final results: '
        for key, value in final_eval.items():
            eval_log += f'{key}: {value:.4f} '
        print(eval_log)

if __name__ == '__main__':
    main()
