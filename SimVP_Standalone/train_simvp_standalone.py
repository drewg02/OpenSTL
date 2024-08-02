import os.path as osp
import time
import warnings
warnings.filterwarnings('ignore')

from simvp_standalone.experiment_recorder import generate_unique_id
from simvp_standalone.simvp_experiment import SimVP_Experiment
from simvp_standalone.simvp_utils import create_parser, get_dist_info, generate_config, update_config, load_config, setup_multi_processes, format_seconds


try:
    import nni

    has_nni = True
except ImportError:
    has_nni = False

if __name__ == '__main__':
    start_time = time.time()

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

    if not config.get('ex_name'):
        config['ex_name'] = generate_unique_id(config)

    if not config.get('ex_dir'):
        config['ex_dir'] = f'{config["ex_name"]}'

    if not config.get('datafile_in'):
        raise ValueError('datafile_in is required')

    setup_multi_processes(config)

    exp = SimVP_Experiment(args)
    rank, world_size = get_dist_info()

    if args.dist:
        print(f"Dist info: rank={rank}, world_size={world_size}")

    print('>' * 35 + f' training {args.ex_name} ' + '<' * 35)
    exp.train()

    print('>' * 35 + f' testing {args.ex_name}  ' + '<' * 35)
    eval_res, _ = exp.test()

    if rank == 0 and has_nni and 'mse' in eval_res:
        nni.report_final_result(eval_res['mse'])

    if rank == 0:
        elapsed_time = time.time() - start_time
        print(f'Total time: {format_seconds(elapsed_time)}')
