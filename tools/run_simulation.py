from openstl.api import BaseExperiment
from openstl.utils import default_parser
from openstl.simulation.utils import create_parser, load_data_loaders, generate_configs

def main():
    args = create_parser().parse_args()

    if not args.train and not args.test:
        raise ValueError("At least one of the 'train' or 'test' flags must be provided.")

    dataloader_train, dataloader_val, dataloader_test = load_data_loaders(args.datafolder_in,
                                                                          args.pre_seq_length,
                                                                          args.aft_seq_length,
                                                                          args.batch_size,
                                                                          args.val_batch_size,
                                                                          args.val_batch_size)

    image_height, image_width = next(iter(dataloader_train))[0].shape[-2:]
    custom_training_config, custom_model_config = generate_configs(args.pre_seq_length, args.aft_seq_length,
                                                                   image_height, image_width, args)

    config = args.__dict__

    config.update(custom_training_config)
    config.update(custom_model_config)
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    exp = BaseExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test))
    
    if args.train:
        print('>' * 35 + f' training {args.ex_name} ' + '<' * 35)
        exp.train()
    if args.test:
        print('>' * 35 + f' testing {args.ex_name}  ' + '<' * 35)
        exp.test()

if __name__ == '__main__':
    main()
