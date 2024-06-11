# Copyright (c) CAIRI AI Lab. All rights reserved

import json
import logging
import os
import os.path as osp
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist

from openstl.api import BaseExperiment
from openstl.core import metric, Recorder, get_priority, hook_maps
from openstl.methods import method_maps
from openstl.utils import (set_seed, print_log, check_dir, collect_env,
                           init_dist, init_random_seed,
                           get_dataset, get_dist_info, weights_to_cpu)

try:
    import nni

    has_nni = True
except ImportError:
    has_nni = False


class SimulationExperiment(BaseExperiment):
    """Extension of the default OpenSTL BaseExperiment."""

    def __init__(self, args, dataloaders=None):
        """Initialize experiments (non-dist as an example)"""
        super().__init__(args, dataloaders)

    def _acquire_device(self):
        """Setup devices"""
        self._use_gpu = self.args.use_gpu and torch.cuda.is_available()

        if self.args.dist and not self._use_gpu:
            assert False, "Distributed training requires GPUs"

        if self._use_gpu:
            device = f'cuda:{self.args.local_rank if self.args.dist else 0}'
            if self.args.dist:
                torch.cuda.set_device(self.args.local_rank)
                print_log(f'Use distributed mode with GPUs: local rank={self.args.local_rankk}')
            else:
                print_log(f'Use non-distributed mode with GPU: {device}')
        else:
            device = 'cpu'
            print_log('No GPU available, defaulting to CPU' if self.args.use_gpu else 'Use CPU')

        return torch.device(device)

    def _preparation(self, dataloaders=None):
        """Preparation of environment and basic experiment setups"""
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(self.args.local_rank)

        # init distributed env first, since logger depends on the dist info.
        if self.args.launcher != 'none' or self.args.dist:
            self._dist = True
        if self._dist:
            assert self.args.launcher != 'none'
            dist_params = dict(backend='nccl', init_method='env://')
            if self.args.launcher == 'slurm':
                dist_params['port'] = self.args.port
            init_dist(self.args.launcher, **dist_params)
            self._rank, self._world_size = get_dist_info()
            # re-set gpu_ids with distributed training mode
            self._gpu_ids = range(self._world_size)
        self.device = self._acquire_device()
        if self._early_stop <= self._max_epochs // 5:
            self._early_stop = self._max_epochs * 2

        # log and checkpoint
        base_dir = self.args.res_dir if self.args.res_dir is not None else 'work_dirs'
        self.path = osp.join(base_dir, self.args.ex_name if not self.args.ex_name.startswith(self.args.res_dir) \
            else self.args.ex_name.split(self.args.res_dir + '/')[-1])
        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        if self._rank == 0:
            check_dir(self.path)
            check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        if self._rank == 0:
            with open(sv_param, 'w') as file_obj:
                json.dump(self.args.__dict__, file_obj)

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            prefix = 'train' if (not self.args.test and not self.args.inference) else 'test'
            logging.basicConfig(level=logging.INFO,
                                filename=osp.join(self.path, '{}_{}.log'.format(prefix, timestamp)),
                                filemode='a', format='%(asctime)s - %(message)s')

        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        if self._rank == 0:
            print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # set random seeds
        if self._dist:
            seed = init_random_seed(self.args.seed)
            seed = seed + dist.get_rank() if self.args.diff_seed else seed
        else:
            seed = self.args.seed
        set_seed(seed)

        # prepare data
        self._get_data(dataloaders)
        # build the method
        self._build_method()
        # build hooks
        self._build_hook()
        # resume traing
        if self.args.auto_resume:
            self.args.resume_from = osp.join(self.checkpoints_path, 'latest.pth')
        if self.args.resume_from is not None:
            self._load(name=self.args.resume_from)
        self.call_hook('before_run')

    def _build_method(self):
        self.steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, self.steps_per_epoch)
        self.method.model.eval()
        # setup ddp training
        if self._dist:
            self.method.model.cuda()
            if self.args.torchscript:
                self.method.model = torch.jit.script(self.method.model)
            self.method._init_distributed()

    def _build_hook(self):
        for k in self.args.__dict__:
            if k.lower().endswith('hook'):
                hook_cfg = self.args.__dict__[k].copy()
                priority = get_priority(hook_cfg.pop('priority', 'NORMAL'))
                hook = hook_maps[k.lower()](**hook_cfg)
                if hasattr(hook, 'priority'):
                    raise ValueError('"priority" is a reserved attribute for hooks')
                hook.priority = priority  # type: ignore
                # insert the hook to a sorted list
                inserted = False
                for i in range(len(self._hooks) - 1, -1, -1):
                    if priority >= self._hooks[i].priority:  # type: ignore
                        self._hooks.insert(i + 1, hook)
                        inserted = True
                        break
                if not inserted:
                    self._hooks.insert(0, hook)

    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders"""
        if dataloaders is None:
            self.train_loader, self.vali_loader, self.test_loader = \
                get_dataset(self.args.dataname, self.config)
        else:
            self.train_loader, self.vali_loader, self.test_loader = dataloaders

        if self.vali_loader is None:
            self.vali_loader = self.test_loader

        if self.args.test:
            self.train_loader = self.test_loader

        self._max_iters = self._max_epochs * len(self.train_loader)

    def _save(self, name=''):
        """Saving models and meta data to checkpoints"""
        checkpoint = {
            'epoch': self._epoch + 1,
            'optimizer': self.method.model_optim.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()) \
                if not self._dist else weights_to_cpu(self.method.model.module.state_dict()),
            'scheduler': self.method.scheduler.state_dict()}
        torch.save(checkpoint, osp.join(self.checkpoints_path, name + '.pth'))

    def _load(self, name=''):
        """Loading models from the checkpoint"""
        filename = name if osp.isfile(name) else osp.join(self.checkpoints_path, name + '.pth')
        try:
            checkpoint = torch.load(filename)
        except:
            return
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
        self._load_from_state_dict(checkpoint['state_dict'])
        if checkpoint.get('epoch', None) is not None:
            self._epoch = checkpoint['epoch']
            self.method.model_optim.load_state_dict(checkpoint['optimizer'])
            self.method.scheduler.load_state_dict(checkpoint['scheduler'])

    def _load_from_state_dict(self, state_dict):
        if self._dist:
            try:
                self.method.model.module.load_state_dict(state_dict)
            except:
                self.method.model.load_state_dict(state_dict)
        else:
            self.method.model.load_state_dict(state_dict)

    def train(self):
        """Training loops of STL methods"""
        start_time = time.time()

        recorder = Recorder(verbose=True, early_stop_time=min(self._max_epochs // 10, 10))
        num_updates = self._epoch * self.steps_per_epoch
        early_stop = False
        self.call_hook('before_train_epoch')

        eta = 1.0  # PredRNN variants

        results = {
            'train_loss': np.zeros(self._max_epochs),
            'vali_loss': np.zeros(self._max_epochs),
            'lr': np.zeros(self._max_epochs)
        }

        for epoch in range(self._epoch, self._max_epochs):
            if self._dist and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            num_updates, loss_mean, eta = self.method.train_one_epoch(self, self.train_loader,
                                                                      epoch, num_updates, eta)

            self._epoch = epoch
            if epoch % self.args.log_step == 0:
                cur_lr = self.method.current_lr()
                cur_lr = sum(cur_lr) / len(cur_lr)
                with torch.no_grad():
                    vali_loss = self.vali()

                if self._rank == 0:
                    print_log(
                        'Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                            epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg, vali_loss))
                    early_stop = recorder(vali_loss, self.method.model, self.path)

                    # retain data
                    results['train_loss'][epoch] = loss_mean.avg
                    results['vali_loss'][epoch] = vali_loss
                    results['lr'][epoch] = cur_lr

                    # save checkpoints
                    if self.args.checkpoint_interval is not None and (epoch + 1) % self.args.checkpoint_interval == 0:
                        self._save(name=f'epoch_{epoch + 1}')

                        best_model_path = osp.join(self.path, 'checkpoint.pth')
                        if osp.exists(best_model_path):
                            new_save_path = osp.join(self.checkpoints_path, f'epoch_best_{epoch + 1}.pth')
                            shutil.copy(best_model_path, new_save_path)

            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()
            if epoch > self._early_stop and early_stop:  # early stop training
                print_log('Early stop training at f{} epoch'.format(epoch))

        if not check_dir(self.path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self._load_from_state_dict(torch.load(best_model_path))
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

        print_log(f'Training time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

        if self._rank == 0:
            folder_path = osp.join(self.path, 'saved')
            check_dir(folder_path)

            for np_data in results.keys():
                np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])

    def test(self, save_dir=None, model_path=None):
        """A testing loop of STL methods with an optional model path for loading."""
        if model_path:
            # Check if the path is a state_dict or a full checkpoint and load accordingly
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # If it's a checkpoint dictionary, load the necessary components
                self._load_from_state_dict(checkpoint['state_dict'])
                if 'optimizer' in checkpoint and 'scheduler' in checkpoint:
                    self.method.model_optim.load_state_dict(checkpoint['optimizer'])
                    self.method.scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                # Assume the file is a state_dict
                self._load_from_state_dict(checkpoint)
        else:
            # Default behavior: load the best model
            best_model_path = os.path.join(self.path, 'checkpoint.pth')
            self._load_from_state_dict(torch.load(best_model_path))

        self.call_hook('before_val_epoch')
        results = self.method.test_one_epoch(self, self.test_loader)
        self.call_hook('after_val_epoch')

        if 'weather' in self.args.dataname:
            metric_list, spatial_norm = self.args.metrics, True
            channel_names = self.test_loader.dataset.data_name if 'mv' in self.args.dataname else None
        else:
            metric_list, spatial_norm, channel_names = self.args.metrics, False, None
        eval_res, eval_log = metric(results['preds'], results['trues'],
                                    self.test_loader.dataset.mean, self.test_loader.dataset.std,
                                    metrics=metric_list, channel_names=channel_names, spatial_norm=spatial_norm)
        results['metrics'] = np.array(eval_res)

        if self._rank == 0:
            print_log(eval_log)
            folder_path = save_dir if save_dir else osp.join(self.path, 'saved')
            check_dir(folder_path)

            for np_data in ['metrics']:
                np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])

            for result_data in ['inputs', 'trues', 'preds']:
                data = results[result_data]
                for i in range(len(data)):
                    line = data[i]

                    unique_id = self.test_loader.dataset.data['samples'][i][0].split('/')[-2]

                    save_path = osp.join(folder_path, result_data, unique_id)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)

                    for j in range(len(line)):
                        np.save(osp.join(save_path, f'{j}.npy'), line[j])

        return eval_res['mse']
