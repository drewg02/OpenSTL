from openstl.api import BaseExperiment
from openstl.utils import default_parser
from openstl.simulation.utils import create_parser, load_data, create_dataloaders, generate_configs

def main():
    args = create_parser().parse_args()

    if not args.train and not args.test:
        raise ValueError("At least one of the 'train' or 'test' flags must be provided.")

    dataset = load_data(args.datafile_in)
    dataloader_train, dataloader_val, dataloader_test = create_dataloaders(dataset, args.batch_size)

    pre_seq_length, aft_seq_length = dataset['X_train'].shape[1], dataset['Y_train'].shape[1]
    image_height, image_width = dataset['X_train'].shape[3], dataset['X_train'].shape[4]
    custom_training_config, custom_model_config = generate_configs(pre_seq_length, aft_seq_length,
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
