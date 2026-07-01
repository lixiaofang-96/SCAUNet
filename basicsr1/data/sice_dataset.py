import os
import random
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

import numpy as np

from basicsr1.utils import FileClient, imfrombytes, img2tensor, padding
from basicsr1.data.transforms import random_augmentation


class Dataset_SICE(data.Dataset):
    """
    Folder-based exposure dataset with a separate Label folder.

    Directory structure:
        dataroot/
            scene_001/
                1.jpg
                2.jpg
                ...
            scene_002/
                ...
            Label/
                scene_001.jpg
                scene_002.jpg
                ...

    Training:
        - traverse all images in all scene folders
        - each image is virtually repeated `repeat_per_image` times
        - each access performs random crop / augmentation

    Testing:
        - can evaluate all images in each folder
        - or fixed one image per folder
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt['phase']
        self.root = opt['dataroot']
        self.io_backend_opt = opt['io_backend'].copy()
        self.file_client = None

        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.geometric_augs = opt.get('geometric_augs', False)

        # new: each image is repeated multiple times in training
        self.repeat_per_image = opt.get('repeat_per_image', 4)

        # test mode: 'all', 'random', 'middle'
        self.test_mode = opt.get('test_mode', 'all')

        self.scene_dirs, self.label_dir = self._scan_root(self.root)

        if self.phase == 'train':
            self.samples = self._build_train_index()
        else:
            self.samples = self._build_test_index()

        print(f'[{self.__class__.__name__}] phase={self.phase}, num_samples={len(self.samples)}')

    def _is_image_file(self, fname):
        return fname.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        )

    def _scan_root(self, root):
        label_dir = osp.join(root, 'Label')
        assert osp.isdir(label_dir), f'Label folder not found in {root}'

        scene_dirs = []
        for name in sorted(os.listdir(root)):
            full = osp.join(root, name)
            if osp.isdir(full) and name != 'Label':
                scene_dirs.append(full)

        return scene_dirs, label_dir

    def _find_label_path(self, scene_name):
        candidates = [
            osp.join(self.label_dir, scene_name + ext)
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG']
        ]
        for p in candidates:
            if osp.isfile(p):
                return p
        raise FileNotFoundError(f'Cannot find GT for scene {scene_name} in {self.label_dir}')

    def _list_scene_images(self, scene_dir):
        imgs = [
            osp.join(scene_dir, f)
            for f in sorted(os.listdir(scene_dir))
            if self._is_image_file(f)
        ]
        return imgs

    def _build_train_index(self):
        """
        Traverse all images in all scene folders.
        Each image is virtually repeated `repeat_per_image` times.
        """
        samples = []
        for scene_dir in self.scene_dirs:
            scene_name = osp.basename(scene_dir)
            gt_path = self._find_label_path(scene_name)
            img_list = self._list_scene_images(scene_dir)

            for lq_path in img_list:
                for _ in range(self.repeat_per_image):
                    samples.append({
                        'scene_dir': scene_dir,
                        'scene_name': scene_name,
                        'gt_path': gt_path,
                        'lq_path': lq_path,
                        'mode': 'train'
                    })
        return samples

    def _build_test_index(self):
        samples = []
        for scene_dir in self.scene_dirs:
            scene_name = osp.basename(scene_dir)
            gt_path = self._find_label_path(scene_name)
            img_list = self._list_scene_images(scene_dir)

            if self.test_mode == 'all':
                for lq_path in img_list:
                    samples.append({
                        'scene_dir': scene_dir,
                        'scene_name': scene_name,
                        'gt_path': gt_path,
                        'lq_path': lq_path,
                        'mode': 'test'
                    })
            elif self.test_mode == 'middle':
                idx = len(img_list) // 2
                samples.append({
                    'scene_dir': scene_dir,
                    'scene_name': scene_name,
                    'gt_path': gt_path,
                    'lq_path': img_list[idx],
                    'mode': 'test'
                })
            elif self.test_mode == 'random':
                lq_path = random.choice(img_list)
                samples.append({
                    'scene_dir': scene_dir,
                    'scene_name': scene_name,
                    'gt_path': gt_path,
                    'lq_path': lq_path,
                    'mode': 'test'
                })
            else:
                raise ValueError(f'Unsupported test_mode: {self.test_mode}')

        return samples

    def _read_img(self, path, key):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt
            )

        img_bytes = self.file_client.get(path, key)
        img = imfrombytes(img_bytes, float32=True)
        return img

    def __getitem__(self, index):
        sample = self.samples[index]

        scene_name = sample['scene_name']
        gt_path = sample['gt_path']
        lq_path = sample['lq_path']

        img_gt = self._read_img(gt_path, 'gt')
        img_lq = self._read_img(lq_path, 'lq')

        if self.phase == 'train':
            gt_size = self.opt['gt_size']

            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            h, w, _ = img_gt.shape
            if h > gt_size and w > gt_size:
                top = np.random.randint(0, h - gt_size + 1)
                left = np.random.randint(0, w - gt_size + 1)
                img_gt = img_gt[top:top + gt_size, left:left + gt_size, :]
                img_lq = img_lq[top:top + gt_size, left:left + gt_size, :]

            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'scene': scene_name
        }

    def __len__(self):
        return len(self.samples)