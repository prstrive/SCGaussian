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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import random
import torch.nn as nn

def get_smooth_loss(depth, guide=None):
    grad_disp_x = torch.abs(depth[:, :-1] - depth[:, 1:])
    grad_disp_y = torch.abs(depth[:-1, :] - depth[1:, :])
    
    if guide is None:
        guide = torch.ones_like(depth).detach()
    
    if len(guide.shape)==3:
        grad_img_x = torch.abs(guide[:, :, :-1] - guide[:, :, 1:]).mean(dim=0)
        grad_img_y = torch.abs(guide[:, :-1, :] - guide[:, 1:, :]).mean(dim=0)
    else:
        grad_img_x = torch.abs(guide[:, :-1] - guide[:, 1:])
        grad_img_y = torch.abs(guide[:-1, :] - guide[1:, :])

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    
    smooth_loss = grad_disp_x.mean() + grad_disp_y.mean()
        
    return smooth_loss

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    if mask is not None:
        if len(mask.shape)==2:
            mask = mask.unsqueeze(0)
        mask = F.conv2d(mask, window[:1], padding=window_size // 2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        ssim_map = ssim_map * mask
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def margin_l2_loss(network_output, gt, mask_patches, margin, return_mask=False):
    network_output = network_output[mask_patches]
    gt = gt[mask_patches]
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask

def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def patch_norm_mse_loss(input, target, fore_mask, patch_size, margin=0.2, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    mask_patches = patchify(fore_mask, patch_size).sum(dim=1) < (patch_size*patch_size / 3)
    return margin_l2_loss(input_patches, target_patches, mask_patches, margin, return_mask)


def ranking_loss(input, target, patch_size, margin=1e-4):
    input_patches = patchify(input, patch_size)
    target_patches = patchify(target, patch_size)
    
    rand_idxes = random.sample(list(range(input_patches.shape[1])), 6)
    
    input_pixels = input_patches[:, rand_idxes].reshape(-1, 2)
    target_pixels = target_patches[:, rand_idxes].reshape(-1, 2)
    
    g = target_pixels[:, 0] - target_pixels[:, 1]
    t = input_pixels[:, 0] - input_pixels[:, 1]
    
    t = torch.where(g < 0, t, -t)
    
    t = t + margin
    
    l = torch.mean(t[t>0])
    
    return l

def cons_loss(input, target, patch_size, margin=1e-4):
    input_patches = patchify(input, patch_size)
    target_patches = patchify(target, patch_size)
    
    
    tmp = (target_patches[:, :, None] - target_patches[:, None, :]).abs()
    tmp1 = torch.eye(target_patches.shape[1]).unsqueeze(0).repeat(target_patches.shape[0], 1, 1).type_as(tmp)
    tmp[tmp1>1] = 1e5
    
    sorted_args = torch.argsort(tmp, dim=-1)[:, :, :2]
    tmp_t = torch.gather(tmp, -1, sorted_args)
    t = (input_patches[:, :, None] - input_patches[:, None, :]).abs()
    t = torch.gather(t, -1, sorted_args)
    
    t = t - margin
    
    l = torch.mean(t[(t>0) & (tmp_t<0.01)])
    
    return l
    
    
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(5, 1)
        self.mu_y_pool = nn.AvgPool2d(5, 1)
        self.sig_x_pool = nn.AvgPool2d(5, 1)
        self.sig_y_pool = nn.AvgPool2d(5, 1)
        self.sig_xy_pool = nn.AvgPool2d(5, 1)
        self.mask_pool = nn.AvgPool2d(5, 1)
        self.refl = nn.ReflectionPad2d(2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask=None):
        x = self.refl(x)
        y = self.refl(y)
        if mask is not None:
            mask = self.refl(mask)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        if mask is not None:
            SSIM_mask = self.mask_pool(mask)
            output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        else:
            output = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output
    
def get_pixel_loss(image, gt_image):
    
    l1 = (image - gt_image).abs().mean(dim=0)
    
    ssim_func = SSIM()
    
    ssim_l = ssim_func(image.unsqueeze(0), gt_image.unsqueeze(0)).squeeze(0)
    
    pl = l1 * 0.5 + ssim_l.mean(dim=0) * 0.5
    
    return pl
    
    
def get_virtual_warp_loss(virtual_img, virtual_depth, vir_c2w, intrs, w2cs, img_colors, vir_mask):
    height, width = virtual_img.shape[1:]
    virtual_c2w = torch.eye(4, 4).type_as(w2cs)
    virtual_c2w[:3, :4] = torch.from_numpy(vir_c2w).type_as(w2cs)
    intr = intrs[0]
    nv = intrs.shape[0]
    
    py, px = torch.meshgrid(torch.arange(height), torch.arange(width))
    px, py = px.reshape(-1).type_as(w2cs), py.reshape(-1).type_as(w2cs)
    
    cam_pts = torch.matmul(intr.inverse()[:3, :3], torch.stack([px, py, torch.ones_like(px)]) * virtual_depth.reshape(1, -1))
    world_pts = torch.matmul(virtual_c2w, torch.cat([cam_pts, torch.ones_like(cam_pts[:1])]))   # 4, npts
    cam_pts = torch.matmul(w2cs, world_pts[None])[:, :3]    # nv, 3, npts
    cam_xyz = torch.matmul(intrs[:, :3, :3], cam_pts)   # nv, 3, npts
    cam_xy = cam_xyz[:, :2] / (cam_xyz[:, 2:] + 1e-8)
    
    norm_x = 2 * cam_xy[:, 0] / (width - 1) - 1
    norm_y = 2 * cam_xy[:, 1] / (height - 1) - 1
    norm_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)  # nv, 1, npts, 2
    
    mask = (norm_grid.abs() <= 1).all(dim=-1).reshape(nv, height, width)
    
    warp_img = F.grid_sample(img_colors, norm_grid, mode="bilinear")
    warp_img = warp_img.reshape(nv, 3, height, width)
    
    l1 = (virtual_img.unsqueeze(0) - warp_img).abs().mean(dim=1)
    
    ssim_func = SSIM()
    
    ssim_l = ssim_func(virtual_img.unsqueeze(0), warp_img).mean(dim=1)
    
    loss = ssim_l
    
    loss[~mask] = 1000
    
    loss = torch.min(loss, dim=0)[0]
    loss[(loss >= 1000) | (~vir_mask.squeeze(0))] = 0.0
    
    return loss.mean()