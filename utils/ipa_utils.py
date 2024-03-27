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
from pathlib import PosixPath
import json


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
def load_train_data(root=''):

    data = []
    directories = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]

    # Now, enumerate over the directories
    for i, dir in enumerate(directories):
        # Load the data from the director
        data.append(load_train_folder(root=root,  folder=dir))

    return data

def load_train_folder(root='', folder='1'):
    prefs = sorted([elem.split('_')[0] for elem in fnmatch.filter(os.listdir(os.path.join(root, folder)),'*_colors_0.png')])
    data = []

    for pref in prefs:

        left = os.path.join(root, folder, pref + '_colors_0.png')
        right = os.path.join(root, folder, pref + '_colors_1.png')
        depth = os.path.join(root, folder, pref + '_depth_0.png')
        segmap = os.path.join(root, folder, pref + '_class_segmaps.png')

        assert os.path.isfile(left)
        assert os.path.isfile(right)
        assert os.path.isfile(depth)
        assert os.path.isfile(segmap)
        data.append([left, right, depth, segmap])

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

def resize_squeeze(image_path, target_width, target_height, object_count=15, grayscale=False, has_table=True):


    if grayscale:
        # Read the image in grayscale
        img = segmap_to_gt(image_path, object_count, has_table=True)
    else:
        img = cv2.imread(image_path)

        # Resize the image
    resized_img = cv2.resize(img, (target_width, target_height))

    return resized_img


def to_image(gt, folder, iterator):

    gt = gt.cpu().detach().numpy()  # Convert tensor to numpy array
    gt = gt.squeeze()  # Remove singleton dimensions if any
    gt = np.clip(gt, 0, 255).astype(np.uint8)  # Clip and convert to uint8

    filename=f'./../data/ipa/converted_gt/gt_{folder}_{iterator}.png'
    # Save the image using OpenCV
    cv2.imwrite(filename, gt)

    return 0
def to_image_pred(pred, folder, iterator, root,sensor):
    pred = pred.cpu().detach().numpy()  # Convert tensor to numpy array

    # Select maximum value for each pixel (assuming highest value is the prediction)
    grayscale_pred = np.max(pred[0], axis=0)  # Maximum along channel axis

    # Clip and convert to uint8 for grayscale image
    grayscale_pred = np.clip(grayscale_pred, 0, 255).astype(np.uint8)
    filename = f'{root}/{sensor}/{folder}/{iterator}_pred_instr.png'
    cv2.imwrite(filename, grayscale_pred)

    return 0
def data_cleaner(root,folder,sensor,iterator):
    filename = f'{root}/{sensor}/{folder}/{iterator}_pred_instr.png'
    try:
        os.unlink(filename)
    except FileNotFoundError:
        print('\nFile does not exist! (anymore?)\n')
    return 0

def segmap_to_gt(path, objectcount, has_table=True):
    """
    Function to convert a segmentation map to ground truth image, assigning unique values to a specified number of objects.

    Args:
        path (str): Path to the segmentation map image.
        objectcount (int): Number of objects to assign unique values in the output.
        has_table (bool, optional): True if the image has a table, False otherwise. Defaults to True.

    Returns:
        np.ndarray: The ground truth image with assigned object values.
    """

    # Load segmentation map as grayscale image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Get unique grayscale values and their pixel counts
    colors, pixelcount = np.unique(img, return_counts=True)

    # Pair unique colors with their pixel counts
    pairs = list(zip(pixelcount, colors))

    # Sort pairs by pixel count in descending order
    pairs.sort(reverse=True)

    # Extract sorted colors without background
    colors_sorted = [pair[1] for pair in pairs]
    if has_table:
        colors_wo_bg = colors_sorted[2:]  # Ignore two background colors
    else:
        colors_wo_bg = colors_sorted[1:]  # Ignore one background color

    # Limit colors to the specified object count
    short_colors = colors_wo_bg[:objectcount]

    # Create a zero-filled mask with the same shape as the image
    mask = np.zeros_like(img)

    # Initialize a variable to hold the intermediate ground truth image
    image_new = 0

    # Iterate through the shortlisted colors
    for i, shade in enumerate(short_colors):
        # Get indices where the image matches the current shade
        pos_indices = img == shade

        # Update the ground truth image using conditional assignment
        image_new += np.where(pos_indices, objectcount - i, mask)

    # Return the final ground truth image
    return image_new

def load_png(path, keys=None):
    #if type(path) is PosixPath:
        #path = path.as_posix()
    #assert os.path.isfile(path), f"File {path} does not exist"
    #assert '.png' in path, f"File {path} is not a png file"

    #data_keys = [key for key in path.keys()]
    #if keys is None:
    #    keys = data_keys
    #else:
    #    assert [key in data_keys for key in keys], f"Invalid keys ({keys}) for data keys: {data_keys}"

    png_data = {}
    for dir in path:
        for img in dir:
            if img.endswith('.png'):
                png_data[img] = np.array(cv2.imread(img))
        #if os.path.join()path[key].dtype.char == 'S':
        #    try:
        #        png_data[key] = json.loads(bytes(np.array(path[key])))[0]
        #    except:
        #        png_data[key] = path[key]
        #else:
        #    png_data[key] = np.array(path[key])

    return png_data

