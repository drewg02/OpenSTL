import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser, show_video_line, show_video_gif_multiple

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        super(CustomDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels


def main():
    file_path = 'data/2dplate/dataset.pkl'
    batch_size = 8

    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset['X_train'], dataset[
        'X_val'], dataset['X_test'], dataset['Y_train'], dataset['Y_val'], dataset['Y_test']

    train_set = CustomDataset(X=X_train, Y=Y_train)
    val_set = CustomDataset(X=X_val, Y=Y_val)
    test_set = CustomDataset(X=X_test, Y=Y_test)
    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    custom_training_config = {
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'batch_size': 16,
        'val_batch_size': 16,
        'epoch': 5,
        'lr': 1e-4,
        'metrics': ['mse', 'mae'],

        'ex_name': 'e3dlstm_2dplate',
        'dataname': '2dplate',
        'in_shape': [10, 1, 64, 64],
    }

    custom_model_config = {
        'method': 'E3DLSTM',
        # reverse scheduled sampling
        'reverse_scheduled_sampling': 0,
        'r_sampling_step_1':  25000,
        'r_sampling_step_2':  50000,
        'r_exp_alpha':  5000,
        # scheduled sampling
        'scheduled_sampling':  1,
        'sampling_stop_iter':  50000,
        'sampling_start_value':  1.0,
        'sampling_changing_rate':  0.00002,
        # model
        'num_hidden': '128,128,128,128',
        'filter_size': 5,
        'stride': 1,
        'patch_size': 4,
        'layer_norm': 0,
    }

    args = create_parser().parse_args([])
    config = args.__dict__

    # update the training config
    config.update(custom_training_config)
    # update the model config
    config.update(custom_model_config)
    # fulfill with default values
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    exp = BaseExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test))

    print('>' * 35 + ' training ' + '<' * 35)
    exp.train()

    print('>' * 35 + ' testing  ' + '<' * 35)
    exp.test()

    inputs = np.load('./work_dirs/e3dlstm_2dplate/saved/inputs.npy')
    preds = np.load('./work_dirs/e3dlstm_2dplate/saved/preds.npy')
    trues = np.load('./work_dirs/e3dlstm_2dplate/saved/trues.npy')

    example_idx = 0
    show_video_line(inputs[example_idx], ncols=10, vmax=0.6, cbar=False, format='png', cmap='coolwarm',
                    out_path='./work_dirs/e3dlstm_2dplate/2dplate_input.png')
    show_video_line(preds[example_idx], ncols=10, vmax=0.6, cbar=False, format='png', cmap='coolwarm',
                    out_path='./work_dirs/e3dlstm_2dplate/2dplate_pred.png')
    show_video_line(trues[example_idx], ncols=10, vmax=0.6, cbar=False, format='png', cmap='coolwarm',
                    out_path='./work_dirs/e3dlstm_2dplate/2dplate_true.png')

    show_video_gif_multiple(inputs[example_idx], trues[example_idx], preds[example_idx], cmap='coolwarm',
                    out_path='./work_dirs/e3dlstm_2dplate/2dplate.gif')

if __name__ == '__main__':
    main()