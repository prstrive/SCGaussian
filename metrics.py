#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import math
import numpy as np

def readImages(renders_dir, gt_dir, mask_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        # render = render.resize(mask.size)
        # gt = gt.resize(mask.size)
        if os.path.exists(mask_dir / fname):
            mask = Image.open(mask_dir / fname)
            mask = mask.resize(gt.size)
            mask = tf.to_tensor(mask).unsqueeze(0)[:, :3, :, :].cuda()
        else:
            mask = torch.ones((1, 3, *gt.size[::-1])).float().cuda()
        mask_bin = (mask == 1.)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda() * mask + (1-mask))
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda() * mask + (1-mask))
        masks.append(mask_bin)
        image_names.append(fname)
    return renders, gts, image_names, masks

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                mask_dir = method_dir / "dtumask"
                renders, gts, image_names, masks = readImages(renders_dir, gt_dir, mask_dir)
                
                ssims = []
                psnrs = []
                lpipss = []
                avgs = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    o_ssim = ssim(renders[idx], gts[idx])
                    # o_psnr = psnr(renders[idx], gts[idx])
                    o_psnr = psnr(renders[idx][masks[idx]][None, ...], gts[idx][masks[idx]][None, ...])
                    o_lpips = lpips(renders[idx], gts[idx], net_type='vgg')
                    o_avg = torch.exp(torch.log(torch.tensor([10**(-o_psnr / 10), math.sqrt(1 - o_ssim), o_lpips])).mean())
                    ssims.append(o_ssim)
                    psnrs.append(o_psnr)
                    lpipss.append(o_lpips)
                    avgs.append(o_avg)
                    

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  AVG: {:>12.7f}".format(torch.tensor(avgs).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "AVG": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "AVG": {name: lp for lp, name in zip(torch.tensor(avgs).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
