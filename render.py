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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.loss_utils import get_pixel_loss
import numpy as np
import matplotlib.cm as cm
import time


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
    x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)

def visualize_cmap(value,
                   weight,
                   colormap,
                   lo=None,
                   hi=None,
                   percentile=99.,
                   curve_fn=lambda x: x,
                   modulus=None,
                   matte_background=True):
    """Visualize a 1D image and a 1D weighting according to some colormap.

    Args:
    value: A 1D image.
    weight: A weight map, in [0, 1].
    colormap: A colormap function.
    lo: The lower bound to use when rendering, if None then use a percentile.
    hi: The upper bound to use when rendering, if None then use a percentile.
    percentile: What percentile of the value map to crop to when automatically
      generating `lo` and `hi`. Depends on `weight` as well as `value'.
    curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
      `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
    matte_background: If True, matte the image over a checkerboard.

    Returns:
    A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(
      value, weight, [50 - percentile / 2, 50 + percentile / 2])

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = np.finfo(np.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
        np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    if colormap:
        colorized = colormap(value)[:, :, :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return colorized

depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)

def visualization(depth, save_path):

    import matplotlib as mpl
    import matplotlib.cm as cm
    from PIL import Image
    
    vmax = np.percentile(depth, 98)
    vmin = depth.min()
    # print(save_path, vmax, vmin)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='turbo')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    im = Image.fromarray(colormapped_im)
    im.save(save_path)
    
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    # normal_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_mask")
    error_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "error_map")
    dtumask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dtumask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(error_map_path, exist_ok=True)
    makedirs(dtumask_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if name=="train" and idx==0:
            render_pkg = render(view, gaussians, pipeline, background, save_color_pcd=True, color_pcd_save_path=os.path.join(model_path, name))
        else:
            render_pkg = render(view, gaussians, pipeline, background)
            
        rendering = render_pkg["render"]
        depth = render_pkg["rendered_depth"]
        
        # print(idx, view.image_name, depth.min(), depth.max())
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        # depth = ((depth - 3.7545) / (33.5699 - 3.7545)).clamp(0.0, 1.0)
        # depth = ((depth - 3.2882) / (64.3477 - 3.2882)).clamp(0.0, 1.0)
        
        dtumask = view.dtumask
        gt = view.original_image[0:3, :, :]
        error_map = get_pixel_loss(rendering, gt)
        torchvision.utils.save_image(error_map, os.path.join(error_map_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        
        if dtumask is not None:
            torchvision.utils.save_image(dtumask, os.path.join(dtumask_path, '{0:05d}'.format(idx) + ".png"))
        
        # depth_est = (1 - depth * render_pkg["rendered_alpha"]).squeeze().cpu().numpy()
        # depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
        # depth_est = torch.as_tensor(depth_est).permute(2,0,1)
        # torchvision.utils.save_image(depth_est, os.path.join(depth_path, 'color_{0:05d}'.format(idx) + ".png"))
        visualization(depth.detach().cpu().numpy()[0], os.path.join(depth_path, 'color_{0:05d}'.format(idx) + ".png"))
    

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_virtual : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_virtual", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_virtual)