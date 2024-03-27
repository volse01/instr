"""
Dataset class.
"""

import cv2
from pathlib import Path
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf
from utils.utils import load_hdf5
import torchvision.transforms.transforms as transforms
import data_io.augmentation as augm
from utils.utils import pekdict
from utils.ipa_utils import load_train_data, load_png
from utils.tensorboard_utils import _convert_disp, _convert_instanceseg, _convert_rgb
import yaml
import os

cv2.setNumThreads(0)


def worker_init_fn(worker_id):
    """
    Bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
    https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class BaseDataset(Dataset):
    def __init__(self, split='train', apply_augmentation=False):
        self.split = split
        self._transform_aug = None
        self.apply_augmentation = apply_augmentation

    def process_rgb(self, im, h=480, w=640):
        """
        Process rgb image
        """
        if isinstance(im, np.ndarray):
            im = ttf.to_pil_image(im)

        if self.apply_augmentation:
            im = self._transform_aug(im)

        if im.size != (w, h):
            im = ttf.resize(im, (h, w), interpolation=Image.LINEAR)

        im = ttf.to_tensor(im)
        im = ttf.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return im

    def process_segmentation_label(self, seg_map):
        seg_map[seg_map > 0] -= 1  # remove table
        return seg_map

    def depth_to_disp(self, depth, baseline, focal_length=541.14):
        depth = np.nan_to_num(depth)
        depth[depth > 10] = 10
        disp = np.zeros_like(depth, dtype=np.float32)
        disp[depth != 0] = (baseline * focal_length) / depth[depth != 0]
        return disp[np.newaxis, ...]


from PIL import Image

class DatasetPNG(BaseDataset):
    """
    Dataloader for PNG files
    """

    def __init__(self, base_path, split='train', apply_augmentation=False):
        super().__init__(split=split, apply_augmentation=apply_augmentation)
        self.base_path = base_path

        self.data_samples = load_train_data(root=base_path)
        self.len = len(self.data_samples)


        # check if any data sample exists
        if self.len == 0:
            print('No data samples found')

        # augmentations
        self._transform_aug = transforms.Compose([
            transforms.RandomApply([augm.SharpnessAugmentation(factor_interval=(0.,2.))],p=0.4),
            transforms.RandomApply([augm.ContrastAugmentation(factor_interval=(0.2,1.))],p=0.3),
            transforms.RandomApply([augm.BrightnessAugmentation(factor_interval=(0.1,1.))],p=0.5),
            transforms.RandomApply([augm.ColorAugmentation(factor_interval=(0.0,0.5))],p=0.3),
            transforms.RandomApply([augm.ChannelShuffle()], p=0.3),
            transforms.RandomApply([augm.GaussianBlur(radius=[1,3])], p=0.2),
            transforms.RandomApply([augm.SaltAndPepperNoise(prob=0.005)], p=0.2),
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        cv2.setNumThreads(0)  # deadlock prevention: https://github.com/pytorch/pytorch/issues/1355

        view_path = self.data_samples[idx]

        extracted_data = load_png(view_path)
        if self.split != 'test':
            with open(os.path.join(self.base_path, 'config.yaml'), 'r') as f:
                tm = yaml.load(f)
                baseline = tm['modules'][-12]['config']['intrinsics']['interocular_distance']

        disp = self.depth_to_disp(extracted_data[2][0], baseline)
        sample = pekdict()
        sample.add(key='color_0', value=self.process_rgb(extracted_data['colors'][0]), tb=_convert_rgb)
        sample.add(key='color_1', value=self.process_rgb(extracted_data['colors'][1]), tb=_convert_rgb)
        sample.add(key='segmap', value=self.process_segmentation_label(extracted_data['segmap']), tb=_convert_instanceseg)
        sample.add(key='disparity', value=disp, tb=_convert_disp)

        return sample
