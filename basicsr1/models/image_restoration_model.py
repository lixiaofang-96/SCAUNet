import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import os
import random
import numpy as np
import cv2

import torch.nn.functional as F
from functools import partial

import glob

from basicsr1.models.archs import define_network
from basicsr1.models.base_model import BaseModel
from basicsr1.utils import get_root_logger, imwrite, tensor2img
loss_module = importlib.import_module('basicsr1.models.losses')
metric_module = importlib.import_module('basicsr1.metrics')

try :
    from torch.cuda.amp import autocast, GradScaler
    load_amp = True
except:
    load_amp = False


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageLLIRModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageLLIRModel, self).__init__(opt)

        # define mixed precision
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')

        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get(
                'mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get(
                'use_identity', False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)  # 如何写 weighted loss 呢？传参构造Loss函数
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        loss_dict = OrderedDict()

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # ============================================================
            # 1) 一大步
            # ============================================================
            state_big = self.net_g(self.lq, state=None, step_size=1.0)

            self.output_big = torch.clamp(state_big['On'], 0, 1)
            self.illu_map_big = torch.clamp(state_big['G'], 0, 1)

            # ============================================================
            # 2) 两小步（共享参数，参与训练，不能 no_grad）
            # ============================================================
            with torch.no_grad():
                state_h1 = self.net_g(self.lq, state=None, step_size=0.5)
                state_h1 = {k: v.detach() for k, v in state_h1.items()}
                self.output_h1 = torch.clamp(state_h1['On'], 0, 1)
                self.illu_map_h1 = torch.clamp(state_h1['G'], 0, 1)

            state_h2 = self.net_g(self.lq, state=state_h1, step_size=0.5)
            self.output_h2 = torch.clamp(state_h2['On'], 0, 1)
            self.illu_map_h2 = torch.clamp(state_h2['G'], 0, 1)

            # ============================================================
            # 3) 重建损失
            #    大步结果、两小步最终结果都监督到 GT
            # ============================================================
            l_big = self.cri_pix(self.illu_map_big, self.output_big, self.lq, self.gt)
            l_half = self.cri_pix(self.illu_map_h2, self.output_h2, self.lq, self.gt)

            dc_big = dark_channel(self.output_big)
            dc_h2 = dark_channel(self.output_h2)
            dc_gt = dark_channel(self.gt)

            l_dc_big = F.l1_loss(dc_big, dc_gt)
            l_dc_half = F.l1_loss(dc_h2, dc_gt)

            l_dc = l_dc_big + l_dc_half

            # ============================================================
            # 4) shortcut 一致性损失
            #    只约束 On 和 G，不加 Z
            # ============================================================
            l_sc_on = F.l1_loss(self.output_big, self.output_h2)
            l_sc_g = F.l1_loss(self.illu_map_big, self.illu_map_h2)
            l_sc = l_sc_on + l_sc_g

            # ============================================================
            # 6) 总损失
            #    这些权重建议先这样用，后面再调
            # ============================================================
            l_pix = (
                    1.0 * l_big +
                    1.0 * l_half +
                    0.2 * l_sc +
                    0.1 * l_dc
            )

            loss_dict['l_big'] = l_big
            loss_dict['l_half'] = l_half
            loss_dict['l_sc'] = l_sc
            loss_dict['l_dc'] = l_dc
            loss_dict['l_pix'] = l_pix

        self.amp_scaler.scale(l_pix).backward()
        self.amp_scaler.unscale_(self.optimizer_g)

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)

        self.amp_scaler.step(self.optimizer_g)
        self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h -
                                          mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                state = self.net_g_ema(img, state=None, step_size=1.0)
            self.output = torch.clamp(state['On'], 0, 1)
            self.illu_map = torch.clamp(state['G'], 0, 1)
        else:
            self.net_g.eval()
            with torch.no_grad():
                state = self.net_g(img, state=None, step_size=1.0)
            self.output = torch.clamp(state['On'], 0, 1)
            self.illu_map = torch.clamp(state['G'], 0, 1)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            illu_img = tensor2img([visuals['illu_map']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    save_illu_path = osp.join(self.opt['path']['visualization'],
                                              img_name,
                                              f'{img_name}_{current_iter}_illu.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_illu_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_illu.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(illu_img, save_illu_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['illu_map'] = self.illu_map.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, param_key='params'):
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)
            net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(
                param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)


def dark_channel(x, kernel_size=15):
    """
    x: [B, C, H, W], range in [0, 1]
    return: [B, 1, H, W]
    """
    # 先对通道取最小
    x_min, _ = torch.min(x, dim=1, keepdim=True)   # [B,1,H,W]

    # 再对局部窗口取最小
    # min pooling 可用 max_pool 对负值实现
    dark = -F.max_pool2d(-x_min, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return dark


class ImageiterModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageiterModel, self).__init__(opt)

        # define mixed precision
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')

        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get(
                'mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get(
                'use_identity', False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)  # 根据pop出来的loss_type找到对应的loss函数
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)  # 如何写 weighted loss 呢？传参构造Loss函数
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        loss_dict = OrderedDict()

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # ============================================================
            # 1) 多次迭代输出
            # ============================================================
            states = self.net_g(self.lq, state=None, step_size=1.0)

            l_pix = 0.0
            l_big_all = 0.0
            l_dc_all = 0.0

            dc_gt = dark_channel(self.gt)

            for s in states:
                output_i = torch.clamp(s['On'], 0, 1)
                illu_i = torch.clamp(s['G'], 0, 1)

                # reconstruction loss
                l_big = self.cri_pix(illu_i, output_i, self.lq, self.gt)

                # dark channel loss
                dc_i = dark_channel(output_i)
                l_dc = F.l1_loss(dc_i, dc_gt)

                # total loss of this stage
                l_stage = 1.0 * l_big + 0.1 * l_dc

                l_pix += l_stage
                l_big_all += l_big
                l_dc_all += l_dc

            # average over all stages
            assert len(states) > 0, 'The returned state list should not be empty.'
            num_states = len(states)
            l_pix = l_pix / num_states
            l_big_all = l_big_all / num_states
            l_dc_all = l_dc_all / num_states

            # final output for logging / visualization
            self.outputg = torch.clamp(states[-1]['On'], 0, 1)
            self.illu_map = torch.clamp(states[-1]['G'], 0, 1)

            loss_dict['l_big_all'] = l_big_all
            loss_dict['l_dc_all'] = l_dc_all
            loss_dict['l_pix_all'] = l_pix

        self.amp_scaler.scale(l_pix).backward()
        self.amp_scaler.unscale_(self.optimizer_g)

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)

        self.amp_scaler.step(self.optimizer_g)
        self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h -
                                          mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # state = self.net_g_ema(img, state=None, step_size=1.0)
                states = self.net_g(img, state=None, step_size=1.0)
                state = states[-1]
            self.output = torch.clamp(state['On'], 0, 1)
            self.illu_map = torch.clamp(state['G'], 0, 1)
        else:
            self.net_g.eval()
            with torch.no_grad():
                states = self.net_g(img, state=None, step_size=1.0)
                state = states[-1]
            self.output = torch.clamp(state['On'], 0, 1)
            self.illu_map = torch.clamp(state['G'], 0, 1)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            illu_img = tensor2img([visuals['illu_map']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    save_illu_path = osp.join(self.opt['path']['visualization'],
                                              img_name,
                                              f'{img_name}_{current_iter}_illu.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_illu_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_illu.png')

                imwrite(sr_img, save_img_path)
                imwrite(illu_img, save_illu_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['illu_map'] = self.illu_map.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        save_filename = f'net_g_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        save_dict = {}

        net = self.get_bare_model(self.net_g)
        state_dict = net.state_dict()
        new_state_dict = OrderedDict()
        for key, param in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            new_state_dict[new_key] = param.detach().cpu()
        save_dict['params'] = new_state_dict

        if self.ema_decay > 0:
            net_ema = self.get_bare_model(self.net_g_ema)
            state_dict_ema = net_ema.state_dict()
            new_state_dict_ema = OrderedDict()
            for key, param in state_dict_ema.items():
                new_key = key[7:] if key.startswith('module.') else key
                new_state_dict_ema[new_key] = param.detach().cpu()
            save_dict['params_ema'] = new_state_dict_ema

        torch.save(save_dict, save_path)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, param_key='params'):
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)

            net = self.net_g
            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(param_key, list) else [param_key]

            assert len(net) == len(param_key), \
                'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()

                new_state_dict = OrderedDict()
                for key, param in state_dict.items():
                    new_key = key[7:] if key.startswith('module.') else key
                    new_state_dict[new_key] = param.detach().cpu()

                save_dict[param_key_] = new_state_dict

            torch.save(save_dict, save_path)

    # def save(self, epoch, current_iter, **kwargs):
    #     if self.ema_decay > 0:
    #         self.save_network([self.net_g, self.net_g_ema],
    #                           'net_g',
    #                           current_iter,
    #                           param_key=['params', 'params_ema'])
    #     else:
    #         self.save_network(self.net_g, 'net_g', current_iter)
    #     self.save_training_state(epoch, current_iter, **kwargs)
    #
    # def save_best(self, best_metric, param_key='params'):
    #     psnr = best_metric['psnr']
    #     cur_iter = best_metric['iter']
    #     save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
    #     exp_root = self.opt['path']['experiments_root']
    #     save_path = os.path.join(
    #         self.opt['path']['experiments_root'], save_filename)
    #
    #     if not os.path.exists(save_path):
    #         for r_file in glob.glob(f'{exp_root}/best_*'):
    #             os.remove(r_file)
    #         net = self.net_g
    #
    #         net = net if isinstance(net, list) else [net]
    #         param_key = param_key if isinstance(
    #             param_key, list) else [param_key]
    #         assert len(net) == len(
    #             param_key), 'The lengths of net and param_key should be the same.'
    #
    #         save_dict = {}
    #         for net_, param_key_ in zip(net, param_key):
    #             net_ = self.get_bare_model(net_)
    #             state_dict = net_.state_dict()
    #             for key, param in state_dict.items():
    #                 if key.startswith('module.'):  # remove unnecessary 'module.'
    #                     key = key[7:]
    #                 state_dict[key] = param.cpu()
    #             save_dict[param_key_] = state_dict
    #
    #         torch.save(save_dict, save_path)