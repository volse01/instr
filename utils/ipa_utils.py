"""
Various prediction utilities.
"""

import numpy as np
import os
import fnmatch
import torch
from PIL import Image
import torchvision.transforms.functional as ttf
import yaml
from yacs.config import CfgNode
import cv2
from model.instr import INSTR
from utils.colormap import get_spaced_colors


YCB_OBJECTS = [
    '003_cracker_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '021_bleach_cleanser',
    '024_bowl',
    '025_mug',
    '035_power_drill',
    '037_scissors',
    '052_extra_large_clamp',
    '061_foam_brick',
]


def stuff_from_state_dict_path(path):
    cfg_path = '/'.join(path.split('/')[:-2]) + '/config.yaml'
    with open(cfg_path, 'r') as f:
        cfg = CfgNode(yaml.load(f))

    net = INSTR(cfg)
    state_dict = torch.load(path)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    rets = net.load_state_dict(state_dict, strict=False)

    print(f"Loaded state dict from {path}: {rets}")
    return cfg, net


def process_im(im, device=torch.device('cuda')):
    im = np.array(im)[:, :, :3]
    im = ttf.to_pil_image(im)
    im = ttf.resize(im, [480, 640], interpolation=Image.LINEAR)
    im = ttf.to_tensor(im)
    im = ttf.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return im.unsqueeze(0).to(device=device)


def process_depth(depth, thresh=10.):
    depth = np.nan_to_num(depth)
    depth[depth > thresh] = thresh
    return torch.from_numpy(depth)


def process_disp(disp):
    return torch.from_numpy(disp.astype(np.float32))


def disp_to_depth(disp, f, b):
    disp = disp.squeeze().cpu().numpy().astype(np.float32)
    depth = np.zeros_like(disp)
    depth[disp != 0] = b * f / disp[disp != 0]
    return depth


def load_data(root=''):
    sensor = 'synthetic'
    data = {
       'synthetic': {}
    }

    folders = sorted(os.listdir(os.path.join(root, sensor)))

    for folder in folders:
        data[sensor][folder] = load_folder(root=root, sensor=sensor, folder=folder)
    return data


def load_folder(root='', sensor='synthetic', folder='1'):
    root = os.path.join(root, sensor)
    prefs = sorted([elem.split('_')[0] for elem in fnmatch.filter(os.listdir(os.path.join(root, folder)),'*_colors_0.png')])
    data = []

    for pref in prefs:

        left = os.path.join(root, folder, pref + '_colors_0.png')
        right = os.path.join(root, folder, pref + '_colors_1.png')
        depth = os.path.join(root, folder, pref + '_depth_0.png')
        gt = os.path.join(root, folder, pref + '_class_segmaps.png')

        assert os.path.isfile(left)
        assert os.path.isfile(right)
        assert os.path.isfile(depth)
        assert os.path.isfile(gt)
        data.append([left, right, depth, gt])

    return data


def overlay_im_with_masks(im, ma, alpha=0.5):
    """
    Overlays an image with corresponding annotations.
    Args:
        im (uint8 np.array): image of shape h, w, 3
        ma (uint8 np array): mask of shape h, w; expects unique integers for object instances
        alpha (float): see cv2.addWeighted() for more information
    Returns:
        uint8 np.array: colorized image of shape h, w, 3
    """

    if ma.max() == 0:
        return im
    colors = get_spaced_colors(50)
    im_col = im.copy()
    for ctr, i in enumerate(np.unique(ma)[1:]):
        a, b = np.where(ma == i)
        if a != []:
            im_col [a, b, :] = colors[ctr]
    im_overlay = im.copy()
    im_overlay = cv2.addWeighted(im_overlay, alpha, im_col, 1 - alpha, 0.0)
    return im_overlay

def resize_keep_centered(image_path, target_width, target_height, grayscale=False):

    if grayscale:
        # Read the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)
    # Get the original image dimensions
    original_height, original_width = img.shape[:2]

    # Calculate the resizing factors for width and height
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # Choose the resizing factor that results in the smaller difference
    if height_ratio < height_ratio:
        resized_width = target_width
        resized_height = int(original_height * width_ratio)
        crop = (resized_height - target_height) // 2

        resized_img = cv2.resize(img, (resized_width, resized_height))
        cropped_img = resized_img[crop:crop + target_height, 0:target_width]

    else:
        resized_width = int(original_width * height_ratio)
        resized_height = target_height
        crop = (resized_width - target_width) // 2

        # Resize the image
        resized_img = cv2.resize(img, (resized_width, resized_height))
        # Crop the image
        cropped_img = resized_img[0:target_height, crop:crop + target_width]

    return cropped_img

def to_image(gt):

    gt = gt.cpu().detach().numpy()  # Convert tensor to numpy array
    gt = gt.squeeze()  # Remove singleton dimensions if any
    gt = np.clip(gt, 0, 255).astype(np.uint8)  # Clip and convert to uint8

    # Save the image using OpenCV
    cv2.imwrite('./output_image.png', gt)

    return 0

''''def to_image_pred(pred):

    pred = pred.cpu().detach().numpy()  # Convert tensor to numpy array
    pred = pred.squeeze()  # Remove singleton dimensions if any
    pred = np.clip(pred, 0, 255).astype(np.uint8)  # Clip and convert to uint8

    # Save the image using OpenCV
    cv2.imwrite('./output_image_pred.png', pred)

    return 0'''''