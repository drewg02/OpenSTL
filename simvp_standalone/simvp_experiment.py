# Copyright (c) CAIRI AI Lab. All rights reserved

import os
import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis, flop_count_table

from simvp_model import SimVP_Model
from simvp_utils import create_dataloaders, AverageMeter, format_seconds, measure_throughput
from simvp_metrics import calc_ssim, calc_mse, calc_mae, calc_rmse, calc_psnr

try:
    import nni

    has_nni = True
except ImportError:
    has_nni = False

class SimVP_Experiment():
    def __init__(self, args, dataloaders=None):
        self.args = args
        self.device = self.args.device

        self.path = self.args.work_dir
        if not osp.exists(self.path):
            os.makedirs(self.path)

        self.model_path = osp.join(self.path, 'simvp_model.pth')
        self.save_dir = osp.join(self.path, 'saved')

        self._acquire_device()
        self._get_data(dataloaders)

        self._max_epochs = self.args.epochs
        self._steps_per_epoch = len(self.train_loader)
        self._total_steps = self._max_epochs * self._steps_per_epoch

        self.args.in_shape = (self.args.pre_seq_length, *next(iter(self.train_loader))[0].shape[2:])
        self.model = SimVP_Model(**self.args.__dict__).to(self.device)

        opt_args = {
                'weight_decay': 0,
            }
        opt_args.update(lr=self.args.lr, weight_decay=self.args.weight_decay)
        if hasattr(self.args, 'opt_eps') and self.args.opt_eps is not None:
            opt_args['eps'] = self.args.opt_eps
        if hasattr(self.args, 'opt_betas') and self.args.opt_betas is not None:
            opt_args['betas'] = self.args.opt_betas

        self.optimizer = torch.optim.Adam(self.model.parameters(), **opt_args)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args.lr,
            total_steps=self._total_steps,
            final_div_factor=getattr(self.args, 'final_div_factor', 1e4))

        self.criterion = nn.MSELoss()

        for key, value in self.args.__dict__.items():
            print(f"{key}: {value}")

        self.display_method_info()

    def _acquire_device(self):
        self._use_gpu = self.args.use_gpu and torch.cuda.is_available()

        if self.args.dist and not self._use_gpu:
            assert False, "Distributed training requires GPUs"

        if self._use_gpu:
            device = f'cuda:{self.args.local_rank if self.args.dist else 0}'
            if self.args.dist:
                torch.cuda.set_device(self.args.local_rank)
                print(f'Use distributed mode with GPUs: local rank={self.args.local_rank}')
            else:
                print(f'Use non-distributed mode with GPU: {device}')
        else:
            device = 'cpu'
            print('No GPU available, defaulting to CPU' if self.args.use_gpu else 'Use CPU')

        return torch.device(device)

    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders"""
        if dataloaders is None:
            self.train_loader, self.vali_loader, self.test_loader = \
                create_dataloaders(self.args.datafile_in,
                                   self.args.pre_seq_length,
                                   self.args.aft_seq_length,
                                   self.args.batch_size,
                                   self.args.val_batch_size,
                                   self.args.val_batch_size,
                                   self.args.dist)
        else:
            if type(dataloaders) is object:
                self.train_loader, self.vali_loader, self.test_loader = dataloaders()
            else:
                self.train_loader, self.vali_loader, self.test_loader = dataloaders

        if self.vali_loader is None:
            self.vali_loader = self.test_loader

        if self.args.test:
            self.train_loader = self.test_loader

        self.original_test_loader = self.test_loader

    def _load(self, load_path):
        model = torch.load(load_path)
        model.to(self.device)

        return model

    def _save(self, model, save_path):
        torch.save(model, save_path)

    def _predict(self, pre_seq_length, aft_seq_length, batch_x):
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length

            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])

            pred_y = torch.cat(pred_y, dim=1)
        else:
            raise ValueError("Invalid pre_seq_length and aft_seq_length")
        return pred_y

    def display_method_info(self):
        """Plot the basic infomation of supported methods"""
        T, C, H, W = self.args.in_shape
        input_dummy = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)

        dash_line = '-' * 80 + '\n'
        info = self.model.__repr__()
        flops = FlopCountAnalysis(self.model, input_dummy)
        flops = flop_count_table(flops)
        fps = measure_throughput(self.model, input_dummy)
        fps = 'Throughputs of {}: {:.3f}\n'.format('SimVP', fps)
        print('Model info:\n' + info+'\n' + flops+'\n' + fps + dash_line)


    def train(self):
        best_loss = float('inf')
        start_time = time.time()
        for epoch in range(self.args.epochs):
            losses_m = AverageMeter()
            val_losses_m = AverageMeter()
            ssims_m = AverageMeter()
            mses_m = AverageMeter()
            maes_m = AverageMeter()
            rmes_m = AverageMeter()
            psnrs_m = AverageMeter()

            self.model.train()

            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                pred_y = self._predict(self.args.pre_seq_length, self.args.aft_seq_length, batch_x)
                loss = self.criterion(pred_y, batch_y)

                losses_m.update(loss.item())

                loss.backward()
                self.optimizer.step()

                torch.cuda.synchronize()

                self.scheduler.step()

            self.model.eval()
            with torch.no_grad():
                for val_batch_x, val_batch_y in self.vali_loader:
                    val_batch_x = val_batch_x.to(self.device)
                    val_batch_y = val_batch_y.to(self.device)

                    val_pred_y = self._predict(self.args.pre_seq_length, self.args.aft_seq_length, val_batch_x)

                    val_loss = self.criterion(val_pred_y, val_batch_y).item()
                    val_losses_m.update(val_loss)

                    ssims_m.update(calc_ssim(val_pred_y.cpu().numpy(), val_batch_y.cpu().numpy()))
                    mses_m.update(calc_mse(val_pred_y.cpu().numpy(), val_batch_y.cpu().numpy()))
                    maes_m.update(calc_mae(val_pred_y.cpu().numpy(), val_batch_y.cpu().numpy()))
                    rmes_m.update(calc_rmse(val_pred_y.cpu().numpy(), val_batch_y.cpu().numpy()))
                    psnrs_m.update(calc_psnr(val_pred_y.cpu().numpy(), val_batch_y.cpu().numpy()))

                if val_losses_m.avg < best_loss:
                    best_loss = val_losses_m.avg

                    print(f'Lowest loss found... Saving best model to {self.model_path}')
                    self._save(self.model, self.model_path)

            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()

            lr = np.mean(np.array([group['lr'] for group in self.optimizer.param_groups]))
            print('Epoch: {0}/{1}, Steps: {2} | Lr: {3:.7f} | Train Loss: {4:.7f} | Vali Loss: {5:.7f}'.format(
                epoch + 1, self._max_epochs, len(self.train_loader), lr, losses_m.avg, val_losses_m.avg))
            print(f"val ssim: {ssims_m.avg}, mse: {mses_m.avg}, mae: {maes_m.avg}, rmse: {rmes_m.avg}, psnr: {psnrs_m.avg}\n")

        elapsed_time = time.time() - start_time
        print(f'Training time: {format_seconds(elapsed_time)}')


    def test(self, save_files=True, do_metrics=True):
        print(f"Loading model from {self.model_path}")
        self._load(self.model_path)

        self.model.eval()

        global ssims_m, mses_m, maes_m, rmes_m, psnrs_m
        if do_metrics:
            ssims_m = AverageMeter()
            mses_m = AverageMeter()
            maes_m = AverageMeter()
            rmes_m = AverageMeter()
            psnrs_m = AverageMeter()
        results = {'inputs': [], 'trues': [], 'preds': []}
        with torch.no_grad():
            for test_batch_x, test_batch_y in self.test_loader:
                test_batch_x = test_batch_x.to(self.device)
                test_batch_y = test_batch_y.to(self.device)

                test_pred_y = self._predict(self.args.pre_seq_length, self.args.aft_seq_length, test_batch_x)

                test_pred_y = test_pred_y.cpu().numpy()
                test_batch_x = test_batch_x.cpu().numpy()
                test_batch_y = test_batch_y.cpu().numpy()

                if do_metrics:
                    ssims_m.update(calc_ssim(test_pred_y, test_batch_x))
                    mses_m.update(calc_mse(test_pred_y, test_batch_x))
                    maes_m.update(calc_mae(test_pred_y, test_batch_x))
                    rmes_m.update(calc_rmse(test_pred_y, test_batch_x))
                    psnrs_m.update(calc_psnr(test_pred_y, test_batch_x))

                results['inputs'].append(test_batch_x)
                results['trues'].append(test_batch_y)
                results['preds'].append(test_pred_y)

        if do_metrics:
            results['metrics'] = {
                'ssim': ssims_m.avg,
                'mse': mses_m.avg,
                'mae': maes_m.avg,
                'rmse': rmes_m.avg,
                'psnr': psnrs_m.avg
            }

        print(f"ssim: {ssims_m.avg}, mse: {mses_m.avg}, mae: {maes_m.avg}, rmse: {rmes_m.avg}, psnr: {psnrs_m.avg}")
        if save_files:
            self.save_results(results, self.save_dir)

        return results['metrics'] if do_metrics else None, {key: results[key] for key in ['inputs', 'trues', 'preds']} if not save_files else None

    def inference(self, save_dir=None, save_files=True):
        return self.test(save_files=save_files, do_metrics=False)

    def save_results(self, results, save_dir=None):
        folder_path = save_dir if save_dir else self.save_dir
        if not osp.exists(folder_path):
            os.makedirs(folder_path)

        if 'metrics' in results:
            np.save(osp.join(folder_path, 'metrics.npy'), results['metrics'])

        for result_data in ['inputs', 'trues', 'preds']:
            assert result_data in results, f"Result data {result_data} not found in results"

            data = results[result_data]
            for i in range(len(data)):
                line = data[i]

                unique_id = self.original_test_loader.dataset.data['samples'][i][0].split('/')[-2]

                save_path = osp.join(folder_path, result_data, unique_id)
                if not osp.exists(save_path):
                    os.makedirs(save_path)

                for j in range(len(line)):
                    file_path = osp.join(str(save_path), f'{j}.npy')
                    np.save(file_path, line[j].reshape(line[j].shape[-2], line[j].shape[-1]))
