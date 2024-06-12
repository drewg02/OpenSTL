from openstl.simulation.train import SimulationExperiment
from openstl.simulation.utils import create_parser, generate_configs
from openstl.utils import default_parser, load_config, update_config, setup_multi_processes


def main():
    args = create_parser().parse_args()

    custom_training_config, custom_model_config = generate_configs(args)

    config = args.__dict__
    config.update(custom_training_config)

    if args.config_file:
        config = update_config(config, load_config(args.config_file))
    else:
        config.update(custom_model_config)
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    setup_multi_processes(config)

    exp = SimulationExperiment(args)

    print('>' * 35 + f' training {args.ex_name} ' + '<' * 35)
    exp.train()

    print('>' * 35 + f' testing {args.ex_name}  ' + '<' * 35)
    exp.test()


if __name__ == '__main__':
    main()
