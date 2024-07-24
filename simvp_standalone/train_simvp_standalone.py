import warnings
warnings.filterwarnings('ignore')

from simvp_experiment import SimVP_Experiment
from experiment_recorder import generate_unique_id

try:
    import nni
    has_nni = True
except ImportError:
    has_nni = False

from argparse import ArgumentParser
def create_parser():
    parser = ArgumentParser(description='SimVP training')
    parser.add_argument('--datafile_in', type=str, default=None, help='Path to the loader file')
    parser.add_argument('--config_file', type=str, default=None, help='Path to the config file')
    parser.add_argument('--ex_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--work_dirs', type=str, default='./work_dirs', help='Working directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--spatio_kernel_enc', type=int, default=3, help='Spatial kernel size for encoder')
    parser.add_argument('--spatio_kernel_dec', type=int, default=3, help='Spatial kernel size for decoder')
    parser.add_argument('--hid_S', type=int, default=64, help='Hidden size for spatial encoder')
    parser.add_argument('--hid_T', type=int, default=512, help='Hidden size for temporal encoder')
    parser.add_argument('--N_T', type=int, default=8, help='Number of temporal layers')
    parser.add_argument('--N_S', type=int, default=4, help='Number of spatial layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--val_batch_size', type=int, default=16, help='Validation batch size')
    parser.add_argument('--drop_path', type=float, default=0, help='Drop path rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--use_gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--dist', action='store_true', help='Use distributed training')
    parser.add_argument('--pre_seq_length', type=int, default=10, help='Length of the input sequence')
    parser.add_argument('--aft_seq_length', type=int, default=10, help='Length of the output sequence')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--clip_grad', type=float, default=0, help='Gradient clipping value')
    parser.add_argument('--clip_mode', type=str, default='norm', help='Gradient clipping mode')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--opt_eps', type=float, default=None, help='Optimizer epsilon')
    parser.add_argument('--opt_betas', type=float, nargs='+', default=None, help='Optimizer betas')
    parser.add_argument('--empty_cache', default=True, action='store_true', help='Empty cache')


    return parser

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

    print('>' * 35 + f' training {args.ex_name} ' + '<' * 35)
    exp = SimVP_Experiment(args)
    exp.train()

    print('>' * 35 + f' testing {args.ex_name}  ' + '<' * 35)
    eval_res, _ = exp.test()

    if has_nni and 'mse' in eval_res:
        nni.report_final_result(eval_res['mse'])
