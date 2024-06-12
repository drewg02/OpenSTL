from openstl.simulation.train import SimulationExperiment
from openstl.simulation.utils import create_parser, create_dataloaders, generate_configs
from openstl.utils import default_parser, load_config, update_config, setup_multi_processes


def main():
    args = create_parser().parse_args()

    dataloader_train, dataloader_val, dataloader_test = create_dataloaders(args.datafile_in,
                                                                           args.pre_seq_length,
                                                                           args.aft_seq_length,
                                                                           args.batch_size,
                                                                           args.val_batch_size,
                                                                           args.val_batch_size,
                                                                           args.dist)

    image_height, image_width = next(iter(dataloader_train))[0].shape[-2:]
    custom_training_config, custom_model_config = generate_configs(args.pre_seq_length, args.aft_seq_length,
                                                                   image_height, image_width, args)

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

    exp = SimulationExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test))

    print('>' * 35 + f' training {args.ex_name} ' + '<' * 35)
    exp.train()

    print('>' * 35 + f' testing {args.ex_name}  ' + '<' * 35)
    exp.test()


if __name__ == '__main__':
    main()
