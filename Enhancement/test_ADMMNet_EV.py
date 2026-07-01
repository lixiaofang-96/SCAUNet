# ADMMNet multi-exposure testing script
# Supports folder-based multi-exposure dataset:
# root/
#   Label/
#       scene1.png
#       scene2.png
#   scene1/
#       1.jpg
#       2.jpg
#       ...
#   scene2/
#       1.jpg
#       2.jpg
#       ...

import os
import argparse
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from skimage.util import img_as_ubyte

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(parent_dir)
sys.path.insert(0, parent_dir)

import utils
from basicsr1.models import create_model
from basicsr1.utils.options import parse


parser = argparse.ArgumentParser(description='ADMMNet multi-exposure testing')

parser.add_argument('--opt', type=str, required=True,
                    help='Path to option YAML file')
parser.add_argument('--weights', type=str, required=True,
                    help='Path to model weights')
parser.add_argument('--gpus', type=str, default="0",
                    help='GPU devices')
parser.add_argument('--dataset', type=str, default='MultiExposure',
                    help='Dataset name for saving results')
parser.add_argument('--result_dir', type=str, default='./results/',
                    help='Directory for results')
parser.add_argument('--GT_mean', action='store_true',
                    help='Use GT mean to rectify output')
parser.add_argument('--self_ensemble', action='store_true',
                    help='Not used here')
args = parser.parse_args()

# =========================
# GPU
# =========================
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

# =========================
# Load option
# =========================
opt = parse(args.opt, is_train=False)
opt['dist'] = False

weights = args.weights
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]

# =========================
# Build model
# =========================
model_restoration = create_model(opt).net_g

checkpoint = torch.load(weights)
try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===> Testing using weights:", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# =========================
# Result directories
# =========================
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')

os.makedirs(result_dir, exist_ok=True)
os.makedirs(result_dir_input, exist_ok=True)
os.makedirs(result_dir_gt, exist_ok=True)

# =========================
# Metrics
# =========================
psnr_big, ssim_big = [], []
psnr_h1, ssim_h1 = [], []
psnr_h2, ssim_h2 = [], []

factor = 4


def rectify_mean(restored, target):
    mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
    mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
    return np.clip(restored * (mean_target / (mean_restored + 1e-8)), 0, 1)


def tensor_to_np_img(x):
    return torch.clamp(x, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()


def is_image_file(name):
    name = name.lower()
    return name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG'))


def find_gt_path(label_dir, scene_name):
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG']:
        candidate = os.path.join(label_dir, scene_name + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def run_three_steps(input_, h, w, model_restoration):
    """
    Returns:
        restored_big, illu_big, z_on_big, z_g_big,
        restored_h1,  illu_h1,  z_on_h1,  z_g_h1,
        restored_h2,  illu_h2,  z_on_h2,  z_g_h2
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


# =========================
# Dataset paths
# =========================
val_root = opt['datasets']['val']['dataroot']
label_dir = os.path.join(val_root, 'Label')
assert os.path.isdir(label_dir), f'Label folder not found: {label_dir}'

scene_dirs = []
for name in sorted(os.listdir(val_root)):
    full = os.path.join(val_root, name)
    if os.path.isdir(full) and name != 'Label':
        scene_dirs.append(full)

print(f"Found {len(scene_dirs)} scene folders.")

# =========================
# Testing
# =========================
with torch.inference_mode():
    for scene_dir in tqdm(scene_dirs, desc='Scenes'):
        scene_name = os.path.basename(scene_dir)
        gt_path = find_gt_path(label_dir, scene_name)
        if gt_path is None:
            print(f'[Skip] GT not found for scene: {scene_name}')
            continue

        target = np.float32(utils.load_img(gt_path)) / 255.

        lq_files = [f for f in natsorted(os.listdir(scene_dir)) if is_image_file(f)]
        if len(lq_files) == 0:
            continue

        os.makedirs(os.path.join(result_dir, scene_name), exist_ok=True)
        os.makedirs(os.path.join(result_dir_input, scene_name), exist_ok=True)
        os.makedirs(os.path.join(result_dir_gt, scene_name), exist_ok=True)

        # 保存一次 GT
        utils.save_img(
            os.path.join(result_dir_gt, scene_name, scene_name + '.png'),
            img_as_ubyte(target)
        )

        for lq_name in lq_files:
            inp_path = os.path.join(scene_dir, lq_name)

            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(inp_path)) / 255.

            # 如果输入和GT尺寸不一致，但只是宽高互换，则旋转输入
            if img.shape[:2] != target.shape[:2]:
                if img.shape[0] == target.shape[1] and img.shape[1] == target.shape[0]:
                    img = np.rot90(img, k=3).copy()

            # 若仍不一致，跳过
            if img.shape[:2] != target.shape[:2]:
                print(f'[Skip shape mismatch] {inp_path} | input={img.shape[:2]} gt={target.shape[:2]}')
                continue

            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img_tensor.unsqueeze(0).cuda()

            # Padding
            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if args.self_ensemble:
                raise NotImplementedError("当前 self_ensemble 仍是旧接口，请先关闭 --self_ensemble")

            if h < 3000 and w < 3000:
                (
                    restored_big, illu_big, z_on_big, z_g_big,
                    restored_h1, illu_h1, z_on_h1, z_g_h1,
                    restored_h2, illu_h2, z_on_h2, z_g_h2
                ) = run_three_steps(input_, h, w, model_restoration)
            else:
                # 大图按列拆分
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

            # Metrics
            psnr_big.append(utils.PSNR(target, restored_big))
            ssim_big.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored_big)))

            psnr_h1.append(utils.PSNR(target, restored_h1))
            ssim_h1.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored_h1)))

            psnr_h2.append(utils.PSNR(target, restored_h2))
            ssim_h2.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored_h2)))

            base_name = os.path.splitext(lq_name)[0]

            save_all_results(
                os.path.join(result_dir, scene_name),
                base_name,
                restored_big, illu_big, z_on_big, z_g_big,
                restored_h1, illu_h1, z_on_h1, z_g_h1,
                restored_h2, illu_h2, z_on_h2, z_g_h2
            )

            # utils.save_img(
            #     os.path.join(result_dir_input, scene_name, base_name + '.png'),
            #     img_as_ubyte(img)
            # )

# =========================
# Average metrics
# =========================
psnr_big = float(np.mean(np.array(psnr_big))) if len(psnr_big) > 0 else 0.0
ssim_big = float(np.mean(np.array(ssim_big))) if len(ssim_big) > 0 else 0.0

psnr_h1 = float(np.mean(np.array(psnr_h1))) if len(psnr_h1) > 0 else 0.0
ssim_h1 = float(np.mean(np.array(ssim_h1))) if len(ssim_h1) > 0 else 0.0

psnr_h2 = float(np.mean(np.array(psnr_h2))) if len(psnr_h2) > 0 else 0.0
ssim_h2 = float(np.mean(np.array(ssim_h2))) if len(ssim_h2) > 0 else 0.0

print("Big: PSNR: %f" % psnr_big)
print("Big: SSIM: %f" % ssim_big)

print("H1 : PSNR: %f" % psnr_h1)
print("H1 : SSIM: %f" % ssim_h1)

print("H2 : PSNR: %f" % psnr_h2)
print("H2 : SSIM: %f" % ssim_h2)