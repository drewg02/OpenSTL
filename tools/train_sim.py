# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from openstl.simulation import SimulationExperiment
from openstl.simulation.utils import create_parser, generate_config
from openstl.utils import (default_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)

try:
    import nni
    has_nni = True
except ImportError:
    has_nni = False


if __name__ == '__main__':
    args = create_parser().parse_args()

    training_config = generate_config(args)

    config = args.__dict__
    config.update(training_config)

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    if args.config_file is None:
        args.config_file = osp.join('./configs', args.dataname, f'{args.method}.py')

    config = update_config(config, load_config(args.config_file),
                           exclude_keys=['method'])

    # set multi-process settings
    setup_multi_processes(config)

    print('>' * 35 + f' training {args.ex_name} ' + '<' * 35)
    exp = SimulationExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>' * 35 + f' testing {args.ex_name}  ' + '<' * 35)
    eval_res, _ = exp.test()

    if rank == 0 and has_nni and 'mse' in eval_res:
        nni.report_final_result(eval_res['mse'])
