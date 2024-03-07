from openstl.api import BaseExperiment
from openstl.utils import default_parser
from openstl.simulation.utils import create_parser, load_data, create_dataloaders, generate_configs

def main():
    args = create_parser().parse_args()

    ex_name = args.ex_name
    file_path = args.datafile
    batch_size = args.batch_size
    pre_seq_length = getattr(args, 'pre_seq_length', 10)
    aft_seq_length = getattr(args, 'aft_seq_length', 10)

    dataset = load_data(file_path)
    dataloader_train, dataloader_val, dataloader_test = create_dataloaders(dataset, batch_size)

    custom_training_config, custom_model_config = generate_configs(ex_name, pre_seq_length, aft_seq_length, batch_size,
                                                                   args)

    config = args.__dict__

    config.update(custom_training_config)
    config.update(custom_model_config)
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    exp = BaseExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test))
    
    if args.train:
        print('>' * 35 + f' training {ex_name} ' + '<' * 35)
        exp.train()
    if args.test:
        print('>' * 35 + f' testing {ex_name}  ' + '<' * 35)
        exp.test()

if __name__ == '__main__':
    main()
