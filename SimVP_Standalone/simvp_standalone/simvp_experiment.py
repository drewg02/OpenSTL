# Copyright (c) CAIRI AI Lab. All rights reserved

import os
import os.path as osp
import time

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.utils import NativeScaler
from tqdm import tqdm

from .simvp_metrics import calc_ssim, calc_mse, calc_mae, calc_rmse, calc_psnr
from .simvp_model import SimVP_Model
from .simvp_utils import create_dataloaders, AverageMeter, format_seconds, measure_throughput, weights_to_cpu, get_dist_info, init_dist, set_seed, init_random_seed

try:
    import nni
    has_nni = True
except ImportError:
    has_nni = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


class SimVP_Experiment():
    _dist = False

    def __init__(self, args, dataloaders=None):
        self.args = args
        self.device = self.args.device
        self.rank = 0
        self.world_size = 1
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

        self.path = os.path.join(self.args.res_dir, self.args.ex_dir)
        if self.rank == 0 and self.local_rank == 0:
            os.makedirs(self.path, exist_ok=True)

        self.model_path = osp.join(self.path, 'simvp_model.pth')
        self.save_dir = osp.join(self.path, 'saved')

        self._preparation(dataloaders)

    def __del__(self):
        if self._dist:
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(f"Failed to destroy process group: {e}")

    def _acquire_device(self):
        """Setup devices"""
        if self.args.device is not None:
            print(f'Use device: {self.args.device}')
            self.device = torch.device(self.args.device)
            self._use_gpu = 'cuda' in self.args.device
            return

        self._use_gpu = self.args.use_gpu and torch.cuda.is_available()

        if self.args.dist and not self._use_gpu:
            assert False, "Distributed training requires GPUs"

        if self._use_gpu and torch.cuda.is_available():
            device = f'cuda:{self.local_rank if self.args.dist else 0}'
            if self.args.dist:
                torch.cuda.set_device(self.local_rank)
                print(f'Use distributed mode with GPUs: local rank={self.local_rank}')
            else:
                print(f'Use non-distributed mode with GPU: {device}')
        else:
            device = 'cpu'
            print('No GPU available, defaulting to CPU' if self.args.use_gpu else 'Use CPU')

        self.device = torch.device(device)

    def _preparation(self, dataloaders=None):
        if self.args.launcher != 'none' or self.args.dist:
            self._dist = True

        if self._dist:
            assert self.args.launcher != 'none'
            dist_params = dict(backend='nccl', init_method='env://')
            if self.args.launcher == 'slurm':
                dist_params['port'] = self.args.port
            init_dist(self.args.launcher, **dist_params)
            self.rank, self.world_size = get_dist_info()

            self._gpu_ids = range(self.world_size)

        if self._dist:
            seed = init_random_seed(self.args.seed)
            seed = seed + dist.get_rank() if self.args.diff_seed else seed
        else:
            seed = self.args.seed
        set_seed(seed)

        self._acquire_device()
        self._get_data(dataloaders)

        self._epoch = None
        self._max_epochs = self.args.epoch
        self._steps_per_epoch = len(self.train_loader)
        self._total_steps = self._max_epochs * self._steps_per_epoch

        self.model = SimVP_Model(**self.args.__dict__).to(self.device)

        if self._dist:
            self.model.cuda()
            if self.args.torchscript:
                self.model = torch.jit.script(self.model)
            self._init_distributed()

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

        if self.rank == 0 and not self.args.no_display_method_info:
            for key, value in self.args.__dict__.items():
                print(f"{key}: {value}")

            self.display_method_info()

    def _init_distributed(self):
        """Initialize DDP training"""
        if self.args.fp16 and has_native_amp:
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()
            if self.rank == 0:
                print('Using native PyTorch AMP. Training in mixed precision (fp16).')
        else:
            print('AMP not enabled. Training in float32.')
        self.model = NativeDDP(self.model, device_ids=[self.local_rank],
                               broadcast_buffers=self.args.broadcast_buffers,
                               find_unused_parameters=self.args.find_unused_parameters)

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

    def _save(self, load_path):
        if not load_path:
            load_path = self.model_path

        saved_model = {
            'epoch': self._epoch + 1,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': weights_to_cpu(self.model.state_dict()) \
                if not self._dist else weights_to_cpu(self.model.module.state_dict()),
            'scheduler': self.scheduler.state_dict()}

        torch.save(saved_model, load_path)

    def _load(self, save_path):
        if not save_path:
            save_path = self.model_path

        try:
            loaded_model = torch.load(save_path)
        except:
            return

        if not isinstance(loaded_model, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {save_path}')
        self._load_from_state_dict(loaded_model['state_dict'])
        if loaded_model.get('epoch', None) is not None:
            self._epoch = loaded_model['epoch']
            self.optimizer.load_state_dict(loaded_model['optimizer'])
            self.scheduler.load_state_dict(loaded_model['scheduler'])

    def _load_from_state_dict(self, state_dict):
        if self._dist:
            try:
                self.model.module.load_state_dict(state_dict)
            except:
                self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

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
        # Throughput requires CUDA
        if self.args.fps and torch.cuda.is_available():
            fps = measure_throughput(self.model, input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format('SimVP', fps)
        else:
            fps = ''
        print('Model info:\n' + info + '\n' + flops + '\n' + fps + dash_line)

    def train(self):
        best_loss = float('inf')
        start_time = time.time()

        for epoch in range(self.args.epoch):
            if self._dist and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            self._epoch = epoch

            losses_m = AverageMeter()
            val_losses_m = AverageMeter()
            ssims_m = AverageMeter()
            mses_m = AverageMeter()
            maes_m = AverageMeter()
            rmes_m = AverageMeter()
            psnrs_m = AverageMeter()

            self.model.train()

            train_pbar = tqdm(self.train_loader) if self.args.pbar and self.rank == 0 and self.local_rank == 0 else self.train_loader

            data_time_m = AverageMeter()
            data_time = time.time()
            for batch_x, batch_y in train_pbar:
                data_time_m.update(time.time() - data_time)

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                pred_y = self._predict(self.args.pre_seq_length, self.args.aft_seq_length, batch_x)
                loss = self.criterion(pred_y, batch_y)

                loss_item = loss.item()
                losses_m.update(loss_item)

                loss.backward()
                self.optimizer.step()

                if self._use_gpu:
                    torch.cuda.synchronize()

                self.scheduler.step()

                if self.rank == 0 and self.local_rank == 0 and self.args.pbar:
                    log_buffer = 'train loss: {:.4f}'.format(loss_item)
                    log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                    train_pbar.set_description(log_buffer)

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

                if self.rank == 0 and val_losses_m.avg < best_loss:
                    best_loss = val_losses_m.avg

                    print(f'Lowest loss found... Saving best model to {self.model_path}')
                    torch.save(self.model.state_dict(), str(self.model_path))
                    if self._dist and self.world_size > 1:
                        dist.barrier()

            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()

            if self.rank == 0:
                lr = np.mean(np.array([group['lr'] for group in self.optimizer.param_groups]))
                print('Epoch: {0}/{1}, Steps: {2} | Lr: {3:.7f} | Train Loss: {4:.7f} | Vali Loss: {5:.7f}'.format(
                    epoch + 1, self._max_epochs, len(self.train_loader), lr, losses_m.avg, val_losses_m.avg))
                print(
                    f"val ssim: {ssims_m.avg}, mse: {mses_m.avg}, mae: {maes_m.avg}, rmse: {rmes_m.avg}, psnr: {psnrs_m.avg}\n")

        if self.rank == 0 and self.local_rank == 0:
            elapsed_time = time.time() - start_time
            print(f'Training time: {format_seconds(elapsed_time)}')

    def test(self, save_files=True, save_dir=None, do_metrics=True):
        print(f"Loading model from {self.model_path}")
        self._load_from_state_dict(torch.load(str(self.model_path)))

        if save_dir is None:
            save_dir = self.save_dir

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
                    ssims_m.update(calc_ssim(test_pred_y, test_batch_y))
                    mses_m.update(calc_mse(test_pred_y, test_batch_y))
                    maes_m.update(calc_mae(test_pred_y, test_batch_y))
                    rmes_m.update(calc_rmse(test_pred_y, test_batch_y))
                    psnrs_m.update(calc_psnr(test_pred_y, test_batch_y))

                results['inputs'].extend(test_batch_x)
                results['trues'].extend(test_batch_y)
                results['preds'].extend(test_pred_y)

        if do_metrics:
            results['metrics'] = {
                'ssim': ssims_m.avg,
                'mse': mses_m.avg,
                'mae': maes_m.avg,
                'rmse': rmes_m.avg,
                'psnr': psnrs_m.avg
            }

        if self.rank == 0:
            print(f"ssim: {ssims_m.avg}, mse: {mses_m.avg}, mae: {maes_m.avg}, rmse: {rmes_m.avg}, psnr: {psnrs_m.avg}")
            if save_files:
                self.save_results(results, save_dir)

        return results['metrics'] if do_metrics else None, {key: results[key] for key in
                                                            ['inputs', 'trues', 'preds']} if not save_files else None

    def inference(self, save_files=True, save_dir=None):
        return self.test(save_files, save_dir, False)

    def save_results(self, results, save_dir=None):
        folder_path = save_dir if save_dir else self.save_dir
        if not osp.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        if 'metrics' in results:
            np.save(osp.join(folder_path, 'metrics.npy'), results['metrics'])

        for result_data in ['inputs', 'trues', 'preds']:
            assert result_data in results, f"Result data {result_data} not found in results"

            data = results[result_data]
            for i in range(len(data)):
                line = data[i]

                unique_id = os.path.split(self.original_test_loader.dataset.data['samples'][i][0])[0].split(os.path.sep)[-1]

                save_path = osp.join(folder_path, result_data, unique_id)
                if not osp.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)

                for j in range(len(line)):
                    file_path = osp.join(str(save_path), f'{j}.npy')
                    np.save(file_path, line[j].reshape(line[j].shape[-2], line[j].shape[-1]))
