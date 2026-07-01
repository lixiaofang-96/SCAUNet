
# Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
# Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, Yulun Zhang
# International Conference on Computer Vision (ICCV), 2023
# https://arxiv.org/abs/2303.06705
# https://github.com/caiyuanhao1998/Retinexformer

from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage.util import img_as_ubyte
from skimage import metrics
import sys

# 获取当前脚本的目录和上级目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(parent_dir)
sys.path.insert(0, parent_dir)

from basicsr1.models import create_model
from basicsr1.utils.options import dict2str, parse
import time

start_time = time.time()

parser = argparse.ArgumentParser(
    description='Image Enhancement using ADMMNet / Retinexformer-style testing')

parser.add_argument('--input_dir', default='./Enhancement/Datasets',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument('--output_dir', default='',
                    type=str, help='Directory for output')
parser.add_argument(
    '--opt', type=str, default='Options/RetinexFormer_SDSD_indoor.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='pretrained_weights/SDSD_indoor.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='SDSD_indoor', type=str,
                    help='Test Dataset')
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true',
                    help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--self_ensemble', action='store_true',
                    help='Use self-ensemble to obtain better results')

args = parser.parse_args()

# 指定 gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False

x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################

model_restoration = create_model(opt).net_g

# 加载模型
checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 生成输出结果的文件
factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
output_dir = args.output_dir

os.makedirs(result_dir, exist_ok=True)
if args.output_dir != '':
    os.makedirs(output_dir, exist_ok=True)

# 三组指标
psnr_big, ssim_big = [], []
psnr_h1, ssim_h1 = [], []
psnr_h2, ssim_h2 = [], []


def rectify_mean(restored, target):
    mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
    mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
    return np.clip(restored * (mean_target / (mean_restored + 1e-8)), 0, 1)


def tensor_to_np_img(x):
    return torch.clamp(x, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()


def run_three_steps(input_, h, w, model_restoration):
    """
    返回:
        restored_big, illu_big, z_on_big, z_g_big,
        restored_h1, illu_h1, z_on_h1, z_g_h1,
        restored_h2, illu_h2, z_on_h2, z_g_h2
    """
    state_big = model_restoration(input_, state=None, step_size=1.0)
    state_h1 = model_restoration(input_, state=None, step_size=0.5)
    state_h2 = model_restoration(input_, state=state_h1, step_size=0.5)

    restored_big = state_big['On'][:, :, :h, :w]
    illu_big = state_big['G'][:, :, :h, :w]
    z_on_big = state_big['Z_on'][:, :, :h, :w]
    z_g_big = state_big['Z_g'][:, :, :h, :w]

    restored_h1 = state_h1['On'][:, :, :h, :w]
    illu_h1 = state_h1['G'][:, :, :h, :w]
    z_on_h1 = state_h1['Z_on'][:, :, :h, :w]
    z_g_h1 = state_h1['Z_g'][:, :, :h, :w]

    restored_h2 = state_h2['On'][:, :, :h, :w]
    illu_h2 = state_h2['G'][:, :, :h, :w]
    z_on_h2 = state_h2['Z_on'][:, :, :h, :w]
    z_g_h2 = state_h2['Z_g'][:, :, :h, :w]

    restored_big = tensor_to_np_img(restored_big)
    illu_big = tensor_to_np_img(illu_big)
    z_on_big = tensor_to_np_img(z_on_big)
    z_g_big = tensor_to_np_img(z_g_big)

    restored_h1 = tensor_to_np_img(restored_h1)
    illu_h1 = tensor_to_np_img(illu_h1)
    z_on_h1 = tensor_to_np_img(z_on_h1)
    z_g_h1 = tensor_to_np_img(z_g_h1)

    restored_h2 = tensor_to_np_img(restored_h2)
    illu_h2 = tensor_to_np_img(illu_h2)
    z_on_h2 = tensor_to_np_img(z_on_h2)
    z_g_h2 = tensor_to_np_img(z_g_h2)

    return (
        restored_big, illu_big, z_on_big, z_g_big,
        restored_h1, illu_h1, z_on_h1, z_g_h1,
        restored_h2, illu_h2, z_on_h2, z_g_h2
    )


def save_all_results(base_dir, base_name,
                     restored_big, illu_big, z_on_big, z_g_big,
                     restored_h1, illu_h1, z_on_h1, z_g_h1,
                     restored_h2, illu_h2, z_on_h2, z_g_h2):
    os.makedirs(base_dir, exist_ok=True)

    # big
    utils.save_img(os.path.join(base_dir, base_name + '_big.png'), img_as_ubyte(restored_big))
    utils.save_img(os.path.join(base_dir, base_name + '_big_illu.png'), img_as_ubyte(illu_big))
    utils.save_img(os.path.join(base_dir, base_name + '_big_z_on.png'), img_as_ubyte(z_on_big))
    utils.save_img(os.path.join(base_dir, base_name + '_big_z_g.png'), img_as_ubyte(z_g_big))

    # h1
    utils.save_img(os.path.join(base_dir, base_name + '_h1.png'), img_as_ubyte(restored_h1))
    utils.save_img(os.path.join(base_dir, base_name + '_h1_illu.png'), img_as_ubyte(illu_h1))
    utils.save_img(os.path.join(base_dir, base_name + '_h1_z_on.png'), img_as_ubyte(z_on_h1))
    utils.save_img(os.path.join(base_dir, base_name + '_h1_z_g.png'), img_as_ubyte(z_g_h1))

    # h2
    utils.save_img(os.path.join(base_dir, base_name + '_h2.png'), img_as_ubyte(restored_h2))
    utils.save_img(os.path.join(base_dir, base_name + '_h2_illu.png'), img_as_ubyte(illu_h2))
    utils.save_img(os.path.join(base_dir, base_name + '_h2_z_on.png'), img_as_ubyte(z_on_h2))
    utils.save_img(os.path.join(base_dir, base_name + '_h2_z_g.png'), img_as_ubyte(z_g_h2))


if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)

    if dataset == 'SID':
        from basicsr1.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset == 'SMID':
        from basicsr1.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    else:
        from basicsr1.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset

    opt_val = opt['datasets']['val']
    opt_val['phase'] = 'test'
    if opt_val.get('scale') is None:
        opt_val['scale'] = 1
    if '~' in opt_val['dataroot_gt']:
        opt_val['dataroot_gt'] = os.path.expanduser('~') + opt_val['dataroot_gt'][1:]
    if '~' in opt_val['dataroot_lq']:
        opt_val['dataroot_lq'] = os.path.expanduser('~') + opt_val['dataroot_lq'][1:]

    dataset_obj = Dataset(opt_val)
    print(f'test dataset length: {len(dataset_obj)}')
    dataloader = DataLoader(dataset=dataset_obj, batch_size=1, shuffle=False)

    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq'].cuda()
            input_save = data_batch['lq'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            target = data_batch['gt'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # Padding
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if args.self_ensemble:
                raise NotImplementedError("当前 self_ensemble 仍是旧接口，请先关闭 --self_ensemble")

            (
                restored_big, illu_big, z_on_big, z_g_big,
                restored_h1, illu_h1, z_on_h1, z_g_h1,
                restored_h2, illu_h2, z_on_h2, z_g_h2
            ) = run_three_steps(input_, h, w, model_restoration)

            if args.GT_mean:
                restored_big = rectify_mean(restored_big, target)
                restored_h1 = rectify_mean(restored_h1, target)
                restored_h2 = rectify_mean(restored_h2, target)

            # 指标
            psnr_big.append(utils.PSNR(target, restored_big))
            ssim_big.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored_big)))

            psnr_h1.append(utils.PSNR(target, restored_h1))
            ssim_h1.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored_h1)))

            psnr_h2.append(utils.PSNR(target, restored_h2))
            ssim_h2.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored_h2)))

            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)

            base_name = os.path.splitext(os.path.split(inp_path)[-1])[0]
            save_all_results(
                os.path.join(result_dir, type_id),
                base_name,
                restored_big, illu_big, z_on_big, z_g_big,
                restored_h1, illu_h1, z_on_h1, z_g_h1,
                restored_h2, illu_h2, z_on_h2, z_g_h2
            )

            utils.save_img(os.path.join(result_dir_input, type_id, base_name + '.png'), img_as_ubyte(input_save))
            utils.save_img(os.path.join(result_dir_gt, type_id, base_name + '.png'), img_as_ubyte(target))

else:
    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val']['dataroot_gt']
    print(input_dir)
    print(target_dir)

    input_paths = natsorted(
        glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.bmp')) )
    target_paths = natsorted(
        glob(os.path.join(target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')) + glob(os.path.join(target_dir, '*.bmp')) )

    with torch.inference_mode():
        for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(inp_path)) / 255.
            target = np.float32(utils.load_img(tar_path)) / 255.

            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding
            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if h < 3000 and w < 3000:
                if args.self_ensemble:
                    raise NotImplementedError("当前 self_ensemble 仍是旧接口，请先关闭 --self_ensemble")

                (
                    restored_big, illu_big, z_on_big, z_g_big,
                    restored_h1, illu_h1, z_on_h1, z_g_h1,
                    restored_h2, illu_h2, z_on_h2, z_g_h2
                ) = run_three_steps(input_, h, w, model_restoration)

            else:
                # 大图按列拆分，仅对 big/h1/h2 分别重组
                input_1 = input_[:, :, :, 1::2]
                input_2 = input_[:, :, :, 0::2]

                (
                    restored_big_1, illu_big_1, z_on_big_1, z_g_big_1,
                    restored_h1_1, illu_h1_1, z_on_h1_1, z_g_h1_1,
                    restored_h2_1, illu_h2_1, z_on_h2_1, z_g_h2_1
                ) = run_three_steps(input_1, h, input_1.shape[3], model_restoration)

                (
                    restored_big_2, illu_big_2, z_on_big_2, z_g_big_2,
                    restored_h1_2, illu_h1_2, z_on_h1_2, z_g_h1_2,
                    restored_h2_2, illu_h2_2, z_on_h2_2, z_g_h2_2
                ) = run_three_steps(input_2, h, input_2.shape[3], model_restoration)

                # 重组
                def merge_half_cols(a1, a2, h, w):
                    out = np.zeros((h, w, a1.shape[2]), dtype=a1.dtype)
                    out[:, 1::2, :] = a1
                    out[:, 0::2, :] = a2
                    return out

                restored_big = merge_half_cols(restored_big_1, restored_big_2, h, w)
                illu_big = merge_half_cols(illu_big_1, illu_big_2, h, w)
                z_on_big = merge_half_cols(z_on_big_1, z_on_big_2, h, w)
                z_g_big = merge_half_cols(z_g_big_1, z_g_big_2, h, w)

                restored_h1 = merge_half_cols(restored_h1_1, restored_h1_2, h, w)
                illu_h1 = merge_half_cols(illu_h1_1, illu_h1_2, h, w)
                z_on_h1 = merge_half_cols(z_on_h1_1, z_on_h1_2, h, w)
                z_g_h1 = merge_half_cols(z_g_h1_1, z_g_h1_2, h, w)

                restored_h2 = merge_half_cols(restored_h2_1, restored_h2_2, h, w)
                illu_h2 = merge_half_cols(illu_h2_1, illu_h2_2, h, w)
                z_on_h2 = merge_half_cols(z_on_h2_1, z_on_h2_2, h, w)
                z_g_h2 = merge_half_cols(z_g_h2_1, z_g_h2_2, h, w)

            if args.GT_mean:
                restored_big = rectify_mean(restored_big, target)
                restored_h1 = rectify_mean(restored_h1, target)
                restored_h2 = rectify_mean(restored_h2, target)

            # 指标
            psnr_big.append(utils.PSNR(target, restored_big))
            ssim_big.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored_big)))

            psnr_h1.append(utils.PSNR(target, restored_h1))
            ssim_h1.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored_h1)))

            psnr_h2.append(utils.PSNR(target, restored_h2))
            ssim_h2.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored_h2)))

            base_name = os.path.splitext(os.path.split(inp_path)[-1])[0]
            save_dir = output_dir if output_dir != '' else result_dir

            save_all_results(
                save_dir,
                base_name,
                restored_big, illu_big, z_on_big, z_g_big,
                restored_h1, illu_h1, z_on_h1, z_g_h1,
                restored_h2, illu_h2, z_on_h2, z_g_h2
            )

# 平均指标
psnr_big = float(np.mean(np.array(psnr_big)))
ssim_big = float(np.mean(np.array(ssim_big)))

psnr_h1 = float(np.mean(np.array(psnr_h1)))
ssim_h1 = float(np.mean(np.array(ssim_h1)))

psnr_h2 = float(np.mean(np.array(psnr_h2)))
ssim_h2 = float(np.mean(np.array(ssim_h2)))

print("Big: PSNR: %f " % psnr_big)
print("Big: SSIM: %f " % ssim_big)

print("H1 : PSNR: %f " % psnr_h1)
print("H1 : SSIM: %f " % ssim_h1)

print("H2 : PSNR: %f " % psnr_h2)
print("H2 : SSIM: %f " % ssim_h2)

end_time = time.time()
print((end_time - start_time) / 15)