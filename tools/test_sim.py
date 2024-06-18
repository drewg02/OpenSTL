# Copyright (c) CAIRI AI Lab. All rights reserved

import warnings
warnings.filterwarnings('ignore')

from openstl.simulation.train import SimulationExperiment
from openstl.simulation.utils import create_parser, generate_config
from openstl.utils import (default_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)

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
    rank, _ = get_dist_info()

    if config['inference'] and not config['test']:
        print('>' * 35 + f' inferencing {args.ex_name}  ' + '<' * 35)
        mse = exp.inference()
    else:
        print('>' * 35 + f' testing {args.ex_name}  ' + '<' * 35)
        mse = exp.test(save_dir=args.saved_path)

    if rank == 0 and has_nni and mse is not None:
        nni.report_final_result(mse)

if __name__ == '__main__':
    main()
