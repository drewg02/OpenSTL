from openstl.simulation.train import SimulationExperiment
from openstl.simulation.utils import create_parser, create_dataloaders, generate_configs
from openstl.utils import default_parser


def main():
    args = create_parser().parse_args()

    dataloader_train, dataloader_val, dataloader_test = create_dataloaders(args.datafile_in,
                                                                           args.pre_seq_length,
                                                                           args.aft_seq_length,
                                                                           args.batch_size,
                                                                           args.val_batch_size,
                                                                           args.val_batch_size)

    image_height, image_width = next(iter(dataloader_train if dataloader_train else dataloader_test))[0].shape[-2:]
    custom_training_config, custom_model_config = generate_configs(args.pre_seq_length, args.aft_seq_length,
                                                                   image_height, image_width, args)

    config = args.__dict__

    config.update(custom_training_config)
    config.update(custom_model_config)
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    exp = SimulationExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test))

    if config['inference'] and not config['test']:
        print('>' * 35 + f' inferencing {args.ex_name}  ' + '<' * 35)
        exp.inference()
    else:
        print('>' * 35 + f' testing {args.ex_name}  ' + '<' * 35)
        exp.test()


if __name__ == '__main__':
    main()
