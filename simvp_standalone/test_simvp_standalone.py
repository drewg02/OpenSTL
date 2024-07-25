import warnings

from .experiment_recorder import generate_unique_id
from .simvp_experiment import SimVP_Experiment
from .simvp_utils import create_parser

warnings.filterwarnings('ignore')

try:
    import nni

    has_nni = True
except ImportError:
    has_nni = False

if __name__ == '__main__':
    args = create_parser().parse_args()

    training_config = {
        'device': 'cuda',
        'spatio_kernel_enc': 3,
        'spatio_kernel_dec': 3,
        'hid_S': 64,
        'hid_T': 512,
        'N_T': 8,
        'N_S': 4,
        'lr': 1e-3,
        'batch_size': 16,
        'drop_path': 0
    }

    config = args.__dict__
    config.update(training_config)

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    if not config.get('ex_name'):
        config['ex_name'] = generate_unique_id(config)

    config['work_dir'] = f'{args.work_dirs}/{config["ex_name"]}'

    if not config.get('datafile_in'):
        raise ValueError('datafile_in is required')

    exp = SimVP_Experiment(args)

    if config['inference'] and not config['test']:
        print('>' * 35 + f' inferencing {args.ex_name}  ' + '<' * 35)
        eval_res, _ = exp.inference()
    else:
        print('>' * 35 + f' testing {args.ex_name}  ' + '<' * 35)
        eval_res, _ = exp.test()

    if has_nni and 'mse' in eval_res:
        nni.report_final_result(eval_res['mse'])
