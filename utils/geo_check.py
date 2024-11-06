import torch
import numpy as np
import cv2
from PIL import Image


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    
    return x_


def get_pairs(c2ws, num_select=10):
    dists = np.linalg.norm(c2ws[:, None, :3, 3] - c2ws[None, :, :3, 3], axis=-1)
    eyes = np.eye(dists.shape[0])
    dists[eyes>0] = 1e3
    sorted_vids = np.argsort(dists, axis=1)
    pairs = sorted_vids[:, :num_select]
    return pairs

def geocheck(intrs, c2ws, depths, dist_thresh=1.0, depth_thresh=0.01, view_thresh=5):
    num_cams = intrs.shape[0]
    num_src = 15
    pairs = get_pairs(c2ws, num_src)
    
    filter_masks = []
    filter_depths = []
    for i in range(num_cams):
        geo_mask_sums = [0] * (num_src-1)
        geo_mask_sum = 0
        depth_est_sum = 0
        depth_ref, intrinsics_ref, extrinsics_ref = depths[i], intrs[i], c2ws[i]
        width, height = depth_ref.shape[1], depth_ref.shape[0]
        x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
        src_ids = pairs[i]
        for j in src_ids:
            depth_src, intrinsics_src, extrinsics_src = depths[j], intrs[j], c2ws[j]
            depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
            
            # check |p_reproj-p_1| < 1
            dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

            # check |d_reproj-d_1| / d_1 < 0.01
            depth_diff = np.abs(depth_reprojected - depth_ref)
            relative_depth_diff = depth_diff / depth_ref

            mask = np.logical_and(dist < dist_thresh, relative_depth_diff < depth_thresh)
            depth_reprojected[~mask] = 0
            
            # mask = None
            # masks = []
            # for im in range(2, 11):
            #     mask = np.logical_and(dist < im / 5, relative_depth_diff < im / 1300)
            #     masks.append(mask)
            # depth_reprojected[~mask] = 0

            geo_mask_sum += mask.astype(np.int32)
            # for im in range(2, num_src+1):
            #     geo_mask_sums[im - 2] += masks[im - 2].astype(np.int32)
            depth_est_sum += depth_reprojected
        
        depth_est_averaged = (depth_est_sum + depth_ref) / (geo_mask_sum + 1)
        fianl_mask = geo_mask_sum > view_thresh
        # fianl_mask = geo_mask_sum >= (num_src+1)
        # for im in range(2, (num_src+1)):
        #     fianl_mask = np.logical_or(fianl_mask, geo_mask_sums[im - 2] >= im)
            
        final_depth = depth_est_averaged * fianl_mask.astype(np.float32)
        filter_masks.append(fianl_mask)
        filter_depths.append(final_depth)
    
    filter_depths = np.stack(filter_depths, axis=0)
    filter_masks = np.stack(filter_masks, axis=0).astype(np.float32)
    
    return filter_depths, filter_masks

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src