"""
Prediction script for STIOS dataset.
Website: https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-17628/#gallery/36367
Download link: https://zenodo.org/record/4706907#.YROCeDqxVhE
Code utilities: https://github.com/DLR-RM/stios-utils
"""

import os
import argparse

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import time
import cv2

from utils.ipa_utils import load_data, process_im, stuff_from_state_dict_path, resize_keep_centered, to_image, to_image_pred
from utils.confmat import ConfusionMatrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict', type=str, default='./pretrained_instr/models/pretrained_model.pth', help="Path to INSTR checkpoint")
    parser.add_argument('--root', type=str, required=True, help="STIOS root")
    args = parser.parse_args()

    assert os.path.isfile(args.state_dict)

    cfg, net = stuff_from_state_dict_path(args.state_dict)

    print('Modifying subpixel correlation layer to fit synthetic data intrinsics')
    time.sleep(1)
    net.adapt_to_new_intrinsics(f_new=585.121 / (1280 / 720), b_new=0.12)

    net = net.cuda().eval()

    paths = load_data(root=args.root)

    results = {}

    for sensor in paths.keys():
        # go through all folders
        for folder in tqdm(paths[sensor].keys()):

            # reset confusion matrix per folder
            ious = []
            f1s = []
            rec = []
            pre = []
            # load images

            for (left, right, depth, gt) in tqdm(paths[sensor][folder]):
                mat = ConfusionMatrix(threshold=0.5)

                left = Image.open(left)
                right = Image.open(right)
                left_t = process_im(left)
                right_t = process_im(right)



                with torch.no_grad():
                    preds = net({'color_0': left_t, 'color_1': right_t})
                pred = preds['predictions_0'][0].unsqueeze(0)


                gt = np.array(resize_keep_centered(gt, pred.shape[3], pred.shape[2]), grayscale=True)
                gt = torch.from_numpy(gt).unsqueeze(0)
                min_index = torch.min(gt)
                gt -= min_index# Shift indices to start from 0
                gt = gt/14
                gt[gt > 15] = 15


                #to_image(gt)
                #to_image_pred(pred)

                mat(pred, targets=gt)
                ious.append(mat.get_iou().item())
                f1s.append(mat.get_f1().item())
                rec.append(mat.get_rec().item())
                pre.append(mat.get_pre().item())

            results[folder] = {
                'ious': ious,
                'f1s': f1s,
                'rec': rec,
                'pre': pre
            }

    mean = []
    print('mIoU:')
    for f, folder in enumerate(results.keys()):
        print(f"{sensor}: {folder}: {np.mean(results[folder]['ious'])*100:.4f}")
        mean.extend(results[folder]['ious'])
    print(f"---\nmean mIoU: {np.mean(mean)*100:.4f}\n")
    mean = []
    print('fi:')
    for f, folder in enumerate(results.keys()):
        print(f"{sensor}: {folder}: {np.mean(results[folder]['f1s'])*100:.4f}")
        mean.extend(results[folder]['f1s'])
    print(f"---\nmean f1: {np.mean(mean)*100:.4f}\n")
    mean = []
    print('Recall:')
    for f, folder in enumerate(results.keys()):
        print(f"{sensor}: {folder}: {np.mean(results[folder]['rec']) * 100:.4f}")
        mean.extend(results[folder]['rec'])
    print(f"---\nmean Recall: {np.mean(mean) * 100:.4f}\n")
    mean = []
    print('Precision:')
    for f, folder in enumerate(results.keys()):
        print(f"{sensor}: {folder}: {np.mean(results[folder]['pre']) * 100:.4f}")
        mean.extend(results[folder]['pre'])
    print(f"---\nmean precision: {np.mean(mean) * 100:.4f}\n")


if __name__ == '__main__':
    main()
