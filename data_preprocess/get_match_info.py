# -*- coding: utf-8 -*-
# @Author  : xuelun

import cv2
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import os
from os.path import join

from dkm.models.model_zoo.DKMv3 import DKMv3

DEFAULT_MIN_NUM_MATCHES = 4
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8
DEFAULT_RANSAC_METHOD = "USAC_MAGSAC"

RANSAC_ZOO = {
    "RANSAC": cv2.RANSAC,
    "USAC_FAST": cv2.USAC_FAST,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
    "USAC_PROSAC": cv2.USAC_PROSAC,
    "USAC_DEFAULT": cv2.USAC_DEFAULT,
    "USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "USAC_ACCURATE": cv2.USAC_ACCURATE,
    "USAC_PARALLEL": cv2.USAC_PARALLEL,
}


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def resize_image(image, size, interp):
    assert interp.startswith('cv2_')
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    # elif interp.startswith('pil_'):
    #     interp = getattr(PIL.Image, interp[len('pil_'):].upper())
    #     resized = PIL.Image.fromarray(image.astype(np.uint8))
    #     resized = resized.resize(size, resample=interp)
    #     resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized


def fast_make_matching_figure(data, b_id):
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    gray0 = cv2.cvtColor(color0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
    kpts0 = data['mkpts0_f'].cpu().numpy()
    kpts1 = data['mkpts1_f'].cpu().numpy()
    mconf = data['mconf'].cpu().numpy()
    inliers = data['inliers']

    rows = 2
    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h = max(h0, h1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w0 + w1

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w0 + margin, margin + w0 + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0[inliers], mkpts1[inliers]):
        c = (0, 255, 0)
        cv2.circle(out, (x0, y0 + sh), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w0, y1 + sh), 3, c, -1, lineType=cv2.LINE_AA)

    return out


def fast_make_matching_overlay(data, b_id):
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    gray0 = cv2.cvtColor(color0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
    kpts0 = data['mkpts0_f'].cpu().numpy()
    kpts1 = data['mkpts1_f'].cpu().numpy()
    mconf = data['mconf'].cpu().numpy()
    inliers = data['inliers']

    rows = 2
    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h = max(h0, h1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w0 + w1

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w0 + margin, margin + w0 + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0[inliers], mkpts1[inliers]):
        c = (0, 255, 0)
        cv2.line(out, (x0, y0 + sh), (x1 + margin + w0, y1 + sh), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0 + sh), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w0, y1 + sh), 3, c, -1, lineType=cv2.LINE_AA)

    return out


def preprocess(image: np.ndarray, grayscale: bool = False, resize_max: int = None,
               dfactor: int = 8):
    image = image.astype(np.float32, copy=False)
    size = image.shape[:2][::-1]
    scale = np.array([1.0, 1.0])

    if resize_max:
        scale = resize_max / max(size)
        if scale < 1.0:
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
            scale = np.array(size) / np.array(size_new)

    if grayscale:
        assert image.ndim == 2, image.shape
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = torch.from_numpy(image / 255.0).float()

    # assure that the size is divisible by dfactor
    size_new = tuple(map(
            lambda x: int(x // dfactor * dfactor),
            image.shape[-2:]))
    image = F.resize(image, size=size_new)
    scale = np.array(size) / np.array(size_new)[::-1]
    return image, scale


def compute_geom(data,
                 ransac_method=DEFAULT_RANSAC_METHOD,
                 ransac_reproj_threshold=DEFAULT_RANSAC_REPROJ_THRESHOLD,
                 ransac_confidence=DEFAULT_RANSAC_CONFIDENCE,
                 ransac_max_iter=DEFAULT_RANSAC_MAX_ITER,
                 ) -> dict:

    mkpts0 = data["mkpts0_f"].cpu().numpy()
    mkpts1 = data["mkpts1_f"].cpu().numpy()

    if len(mkpts0) < 2 * DEFAULT_MIN_NUM_MATCHES:
        return {}

    h1, w1 = data["hw0_i"]

    geo_info = {}

    F, inliers = cv2.findFundamentalMat(
        mkpts0,
        mkpts1,
        method=RANSAC_ZOO[ransac_method],
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=ransac_confidence,
        maxIters=ransac_max_iter,
    )
    if F is not None:
        geo_info["Fundamental"] = F.tolist()

    H, _ = cv2.findHomography(
        mkpts1,
        mkpts0,
        method=RANSAC_ZOO[ransac_method],
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=ransac_confidence,
        maxIters=ransac_max_iter,
    )
    if H is not None:
        geo_info["Homography"] = H.tolist()
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            mkpts0.reshape(-1, 2),
            mkpts1.reshape(-1, 2),
            F,
            imgSize=(w1, h1),
        )
        geo_info["H1"] = H1.tolist()
        geo_info["H2"] = H2.tolist()

    return geo_info


def wrap_images(img0, img1, geo_info, geom_type):
    img0 = img0[0].permute((1, 2, 0)).cpu().numpy()[..., ::-1]
    img1 = img1[0].permute((1, 2, 0)).cpu().numpy()[..., ::-1]

    h1, w1, _ = img0.shape
    h2, w2, _ = img1.shape

    rectified_image0 = img0
    rectified_image1 = None
    H = np.array(geo_info["Homography"])
    F = np.array(geo_info["Fundamental"])

    title = []
    if geom_type == "Homography":
        rectified_image1 = cv2.warpPerspective(
            img1, H, (img0.shape[1], img0.shape[0])
        )
        title = ["Image 0", "Image 1 - warped"]
    elif geom_type == "Fundamental":
        H1, H2 = np.array(geo_info["H1"]), np.array(geo_info["H2"])
        rectified_image0 = cv2.warpPerspective(img0, H1, (w1, h1))
        rectified_image1 = cv2.warpPerspective(img1, H2, (w2, h2))
        title = ["Image 0 - warped", "Image 1 - warped"]
    else:
        print("Error: Unknown geometry type")

    fig = plot_images(
        [rectified_image0.squeeze(), rectified_image1.squeeze()],
        title,
        dpi=300,
    )

    img = fig2im(fig)

    plt.close(fig)

    return img


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, size=5, pad=0.5):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        dpi:
        size:
        pad:
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    figsize = (size * n, size * 6 / 5) if size is not None else None
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)

    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])

    fig.tight_layout(pad=pad)

    return fig


def fig2im(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf_ndarray = np.frombuffer(fig.canvas.tostring_rgb(), dtype="u1")
    im = buf_ndarray.reshape(h, w, 3)
    return im


if __name__ == '__main__':

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = DKMv3(weights=None, h=672, w=896)

    # weights path
    checkpoints_path = join('weights', 'gim_dkm_100h.ckpt')

    # load state dict
    state_dict = torch.load(checkpoints_path, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        if 'encoder.net.fc' in k:
            state_dict.pop(k)

    # load state dict
    model.load_state_dict(state_dict)

    # eval mode
    model = model.eval().to(device)

    # replace to your own path
    root_dir = "/nerf_llff_data/fern/images"
    filenames = sorted(os.listdir(root_dir))
    image_dir = "/nerf_llff_data/fern/images_match"
    os.makedirs(image_dir, exist_ok=True)
    
    # # blendder
    # filenames = [fn for fn in filenames if fn in ['r_2.png', 'r_16.png', 'r_26.png', 'r_55.png', 'r_73.png', 'r_76.png', 'r_86.png', 'r_93.png']]
    
    # tanks and llff
    train_filenames = [c for idx, c in enumerate(filenames) if idx % 8 != 0]
    N_sparse = 3
    idx_train = np.linspace(0, len(train_filenames) - 1, N_sparse)
    idx_train = [round(i) for i in idx_train]
    filenames = [c for idx, c in enumerate(train_filenames) if idx in idx_train]
    
    match_data = {}
    for i in range(len(filenames)-1):
        img_path0 = os.path.join(root_dir, filenames[i])
        if not os.path.isfile(img_path0):
            continue
        img_name0 = os.path.basename(img_path0).split(".")[0]
        if img_name0 not in match_data:
            match_data[img_name0] = {}
        
        image0 = read_image(img_path0)
        image0, scale0 = preprocess(image0)
        image0 = image0.to(device)[None]
        
        for j in range(i+1, len(filenames)):
            img_path1 = os.path.join(root_dir, filenames[j])
            if not os.path.isfile(img_path0):
                continue
            img_name1 = os.path.basename(img_path1).split(".")[0]
            if img_name1 not in match_data:
                match_data[img_name1] = {}
        
            image1 = read_image(img_path1)
            image1, scale1 = preprocess(image1)
            image1 = image1.to(device)[None]
            
            data = dict(color0=image0, color1=image1, image0=image0, image1=image1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dense_matches, dense_certainty = model.match(image0, image1)
                sparse_matches, mconf = model.sample(dense_matches, dense_certainty, 2000)

            height0, width0 = image0.shape[-2:]
            height1, width1 = image1.shape[-2:]

            kpts0 = sparse_matches[:, :2]
            kpts0 = torch.stack((
                width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1,)
            kpts1 = sparse_matches[:, 2:]
            kpts1 = torch.stack((
                width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1,)

            # robust fitting
            _, mask = cv2.findFundamentalMat(kpts0.cpu().numpy(),
                                            kpts1.cpu().numpy(),
                                            cv2.USAC_MAGSAC, ransacReprojThreshold=1.0,
                                            confidence=0.999999, maxIters=10000)
            mask = mask.ravel() > 0
            
            match_data[img_name0][img_name1] = ((sparse_matches[:, :2] + 1) / 2)[mask].cpu().numpy()
            match_data[img_name1][img_name0] = ((sparse_matches[:, 2:] + 1) / 2)[mask].cpu().numpy()
            
            print("match_data num:", match_data[img_name0][img_name1].shape, match_data[img_name1][img_name0].shape)

            b_ids = torch.where(mconf[None])[0]

            data.update({
                'hw0_i': image0.shape[-2:],
                'hw1_i': image1.shape[-2:],
                'mkpts0_f': kpts0,
                'mkpts1_f': kpts1,
                'm_bids': b_ids,
                'mconf': mconf,
                'inliers': mask,
            })

            # save visualization
            alpha = 0.4
            out = fast_make_matching_figure(data, b_id=0)
            overlay = fast_make_matching_overlay(data, b_id=0)
            out = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)
            cv2.imwrite(join(image_dir, f'{img_name0}_{img_name1}_match.png'), out[..., ::-1])
            
            print(f"finish {img_name0}_{img_name1}.")

    np.save(os.path.join(image_dir, "match_data.npy"), match_data)
    
