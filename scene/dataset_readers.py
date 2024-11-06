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

import re
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, read_points3D_binary_pointid
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
import cv2
from utils.virtual_poses import interpolate_virtual_poses_sequential
from utils import pose_utils

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    dtumask: np.array
    blendermask: np.array
    point3D_ids: np.array
    near_far: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    base_cameras: list
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    match_data: dict

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        point3D_ids = extr.point3D_ids

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, f"Colmap camera model not handled for {intr.model}: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # image = image.resize(np.array(image.size) // 8)
        
        cam_info = CameraInfo(uid=extr.camera_id, R=R, T=T, FovY=FovY, FovX=FovX, image=image, point3D_ids=point3D_ids, dtumask=None, near_far=None, blendermask=None,
                              image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1])
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        
        N_sparse = 3
        idx_train = np.linspace(0, len(train_cam_infos) - 1, N_sparse)
        idx_train = [round(i) for i in idx_train]
        other_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx not in idx_train]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_train]
        base_cam_infos = train_cam_infos
        
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _, point_ids = read_points3D_binary_pointid(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
            
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
        
    try:
        xyz, rgb, _, point_ids = read_points3D_binary_pointid(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    
    near_fars = []
    for idx, caminfo in enumerate(train_cam_infos):
        FovX, FovY, R, T, point3D_ids, image, image_name = caminfo.FovX, caminfo.FovY, caminfo.R, caminfo.T, caminfo.point3D_ids, caminfo.image, caminfo.image_name
        
        width, height = image.size[:2]
        fx, fy = fov2focal(FovX, width), fov2focal(FovY, height)
        cx, cy = width / 2.0, height / 2.0
        intr_init = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        w2c_init = np.zeros((4, 4))
        w2c_init[:3, :3] = R.transpose()
        w2c_init[:3, 3] = T
        w2c_init[3, 3] = 1.0
        xyz_view = []
        for i in range(len(point3D_ids)):
            pid = point3D_ids[i]
            if pid!=-1:
                xyz_idx = point_ids[pid]
                xyz_view.append(xyz[xyz_idx])
        xyz_view = np.stack(xyz_view, axis=1)  # 3, npts
        xyz_cam = np.matmul(w2c_init, np.vstack((xyz_view, np.ones_like(xyz_view[:1]))))[:3]
        xyz_pixel = np.matmul(intr_init, xyz_cam)
        colmap_depth = xyz_pixel[2] # npts
        
        near_fars.append([np.min(colmap_depth), np.max(colmap_depth)])
        train_cam_infos[idx] = caminfo._replace(near_far=np.array([np.min(colmap_depth) * 0.8, np.max(colmap_depth) * 1.2]))
    
    # get match data
    all_match_data = np.load(os.path.join(path, "match_data.npy"), allow_pickle=True).item()
    match_data = {}
    for i in range(len(train_cam_infos)-1):
        cam0 = train_cam_infos[i]
        name0 = cam0.image_name
        if name0 not in match_data:
            match_data[name0] = {}
        for j in range(i+1, len(train_cam_infos)):
            cam1 = train_cam_infos[j]
            name1 = cam1.image_name
            if name1 not in match_data:
                match_data[name1] = {}
            
            match_data[name0][name1] = all_match_data[name0][name1]
            match_data[name1][name0] = all_match_data[name1][name0]

    scene_info = SceneInfo(point_cloud=pcd,
                           match_data=match_data,
                           base_cameras = base_cam_infos,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readTanksSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        
        N_sparse = 3
        idx_train = np.linspace(0, len(train_cam_infos) - 1, N_sparse)
        idx_train = [round(i) for i in idx_train]
        other_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx not in idx_train]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_train]
        base_cam_infos = train_cam_infos
        
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _, point_ids = read_points3D_binary_pointid(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
            
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
        
    try:
        xyz, rgb, _, point_ids = read_points3D_binary_pointid(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    
    near_fars = []
    for idx, caminfo in enumerate(train_cam_infos):
        FovX, FovY, R, T, point3D_ids, image, image_name = caminfo.FovX, caminfo.FovY, caminfo.R, caminfo.T, caminfo.point3D_ids, caminfo.image, caminfo.image_name
        
        width, height = image.size[:2]
        fx, fy = fov2focal(FovX, width), fov2focal(FovY, height)
        cx, cy = width / 2.0, height / 2.0
        intr_init = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        w2c_init = np.zeros((4, 4))
        w2c_init[:3, :3] = R.transpose()
        w2c_init[:3, 3] = T
        w2c_init[3, 3] = 1.0
        xyz_view = []
        for i in range(len(point3D_ids)):
            pid = point3D_ids[i]
            if pid!=-1:
                xyz_idx = point_ids[pid]
                xyz_view.append(xyz[xyz_idx])
        xyz_view = np.stack(xyz_view, axis=1)  # 3, npts
        xyz_cam = np.matmul(w2c_init, np.vstack((xyz_view, np.ones_like(xyz_view[:1]))))[:3]
        xyz_pixel = np.matmul(intr_init, xyz_cam)
        colmap_depth = xyz_pixel[2] # npts
        
        near_fars.append([np.min(colmap_depth), np.max(colmap_depth)])
        train_cam_infos[idx] = caminfo._replace(near_far=np.array([np.min(colmap_depth) * 0.8, np.max(colmap_depth) * 1.2]))
    
    # get match data
    all_match_data = np.load(os.path.join(path, "match_data.npy"), allow_pickle=True).item()
    match_data = {}
    for i in range(len(train_cam_infos)-1):
        cam0 = train_cam_infos[i]
        name0 = cam0.image_name
        if name0 not in match_data:
            match_data[name0] = {}
        for j in range(i+1, len(train_cam_infos)):
            cam1 = train_cam_infos[j]
            name1 = cam1.image_name
            if name1 not in match_data:
                match_data[name1] = {}
            
            match_data[name0][name1] = all_match_data[name0][name1]
            match_data[name1][name0] = all_match_data[name1][name0]

    scene_info = SceneInfo(point_cloud=pcd,
                           match_data=match_data,
                           base_cameras = base_cam_infos,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapCamerasDTU(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        point3D_ids = extr.point3D_ids

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, f"Colmap camera model not handled for {intr.model}: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # image = image.resize(np.array(image.size) // 8)
        
        if os.path.exists(os.path.join(images_folder.rsplit("/", 1)[0], "idrmask", "{:0>3}.png".format(int(image_name.split("_")[1])-1))):
            mask = np.array(Image.open(os.path.join(images_folder.rsplit("/", 1)[0], "idrmask", "{:0>3}.png".format(int(image_name.split("_")[1])-1))))
            mask = (np.max(mask, axis=-1) > 10).astype(np.float32)
            
            if (mask.shape[0] != image.size[1]) and (mask.shape[1] != image.size[0]):
                mask = cv2.resize(mask, image.size[:2], interpolation=cv2.INTER_NEAREST)
        else:
            mask = None
        
        cam_info = CameraInfo(uid=extr.camera_id, R=R, T=T, FovY=FovY, FovX=FovX, image=image, point3D_ids=point3D_ids, dtumask=mask, near_far=None, blendermask=None,
                              image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1])
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readDTUSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images
    cam_infos_unsorted = readColmapCamerasDTU(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        
    if eval:
        N_sparse = 3
        train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
        test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
        
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx[:N_sparse]]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
        base_cam_infos = train_cam_infos
        
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _, point_ids = read_points3D_binary_pointid(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
            
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
        
    try:
        xyz, rgb, _, point_ids = read_points3D_binary_pointid(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    
    near_fars = []
    for idx, caminfo in enumerate(train_cam_infos):
        FovX, FovY, R, T, point3D_ids, image, image_name = caminfo.FovX, caminfo.FovY, caminfo.R, caminfo.T, caminfo.point3D_ids, caminfo.image, caminfo.image_name
        
        width, height = image.size[:2]
        fx, fy = fov2focal(FovX, width), fov2focal(FovY, height)
        cx, cy = width / 2.0, height / 2.0
        intr_init = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        w2c_init = np.zeros((4, 4))
        w2c_init[:3, :3] = R.transpose()
        w2c_init[:3, 3] = T
        w2c_init[3, 3] = 1.0
        xyz_view = []
        for i in range(len(point3D_ids)):
            pid = point3D_ids[i]
            if pid!=-1:
                xyz_idx = point_ids[pid]
                xyz_view.append(xyz[xyz_idx])
        xyz_view = np.stack(xyz_view, axis=1)  # 3, npts
        xyz_cam = np.matmul(w2c_init, np.vstack((xyz_view, np.ones_like(xyz_view[:1]))))[:3]
        xyz_pixel = np.matmul(intr_init, xyz_cam)
        colmap_depth = xyz_pixel[2] # npts
        
        near_fars.append([np.min(colmap_depth), np.max(colmap_depth)])
        train_cam_infos[idx] = caminfo._replace(near_far=np.array([np.min(colmap_depth) * 0.8, np.max(colmap_depth) * 1.2]))
    
    # get match data
    all_match_data = np.load(os.path.join(path, "match_data.npy"), allow_pickle=True).item()
    match_data = {}
    for i in range(len(train_cam_infos)-1):
        cam0 = train_cam_infos[i]
        name0 = cam0.image_name
        if name0 not in match_data:
            match_data[name0] = {}
        for j in range(i+1, len(train_cam_infos)):
            cam1 = train_cam_infos[j]
            name1 = cam1.image_name
            if name1 not in match_data:
                match_data[name1] = {}
            
            match_data[name0][name1] = all_match_data[name0][name1]
            match_data[name1][name0] = all_match_data[name1][name0]

    scene_info = SceneInfo(point_cloud=pcd,
                           match_data=match_data,
                           base_cameras = base_cam_infos,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            
            img_mask = norm_data[:, :, 3] > 0

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            
            near_far = np.array([1.0, 6.0])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, near_far=near_far, dtumask=None, blendermask=img_mask, point3D_ids=None,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in [2, 16, 26, 55, 73, 76, 86, 93]]
        eval_cam_infos = [c for idx, c in enumerate(test_cam_infos) if idx % 8 == 0]
        test_cam_infos = test_cam_infos
        base_cam_infos = train_cam_infos
    else:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    # get match data
    all_match_data = np.load(os.path.join(path, "match_data.npy"), allow_pickle=True).item()
    match_data = {}
    for i in range(len(train_cam_infos)-1):
        cam0 = train_cam_infos[i]
        name0 = cam0.image_name
        if name0 not in match_data:
            match_data[name0] = {}
        for j in range(i+1, len(train_cam_infos)):
            cam1 = train_cam_infos[j]
            name1 = cam1.image_name
            if name1 not in match_data:
                match_data[name1] = {}
            
            match_data[name0][name1] = all_match_data[name0][name1]
            match_data[name1][name0] = all_match_data[name1][name0]

    scene_info = SceneInfo(point_cloud=pcd,
                           match_data=match_data,
                           base_cameras = base_cam_infos,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def generateLLFFCameras(poses):
    cam_infos = []
    Rs, tvecs, height, width, focal_length_x = pose_utils.convert_poses(poses) 
    # print(Rs, tvecs, height, width, focal_length_x)
    virtual_poses = []
    for idx, _ in enumerate(Rs):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(Rs)))
        sys.stdout.flush()

        uid = idx
        R = np.transpose(Rs[idx])
        T = tvecs[idx]

        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)
        
        virtual_pose = {
            "R": R,
            "T": T,
            "FovY": FovY,
            "FovX": FovX,
            "uid": uid
        }
        virtual_poses.append(virtual_pose)
        
        # image = Image.fromarray((np.ones((height, width, 3), dtype=np.int32)*255).astype(np.byte), "RGB")

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None, dtumask=None, blendermask=None, point3D_ids=None, near_far=None,
                              image_path=None, image_name=None, width=width, height=height)
        cam_infos.append(cam_info)
        
    np.save("virtual_poses.npy", virtual_poses)
    sys.stdout.write('\n')
    return cam_infos


def CreateLLFFSpiral(basedir):

    # Load poses and bounds.
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses_o = poses_arr[:, :-2].reshape([-1, 3, 5])
    bounds = poses_arr[:, -2:]
    
    # Pull out focal length before processing poses.
    # Correct rotation matrix ordering (and drop 5th column of poses).
    fix_rotation = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
                            dtype=np.float32)
    inv_rotation = np.linalg.inv(fix_rotation)
    poses = poses_o[:, :3, :4] @ fix_rotation

    # Rescale according to a default bd factor.
    # scale = 1. / (bounds.min() * .75)
    # poses[:, :3, 3] *= scale
    # bounds *= scale

    # Recenter poses.
    render_poses = pose_utils.recenter_poses(poses)

    # Separate out 360 versus forward facing scenes.
    render_poses = pose_utils.generate_spiral_path(
          render_poses, bounds, n_frames=180)
    render_poses = pose_utils.backcenter_poses(render_poses, poses)
    render_poses = render_poses @ inv_rotation
    render_poses = np.concatenate([render_poses, np.tile(poses_o[:1, :3, 4:], (render_poses.shape[0], 1, 1))], -1)

    render_cam_infos = generateLLFFCameras(render_poses.transpose([1,2,0]))

    nerf_normalization = getNerfppNorm(render_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=None,
                           base_cameras=None,
                           match_data=None,
                           test_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None)
    return scene_info


def CreateTanksSpiral(path):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos_as = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    cam_infos_ds = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name, reverse=True)
    cam_infos = cam_infos_as + cam_infos_ds

    nerf_normalization = getNerfppNorm(cam_infos)

    
    scene_info = SceneInfo(point_cloud=None,
                           match_data=None,
                           base_cameras = None,
                           train_cameras=None,
                           test_cameras=cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None)
    return scene_info


def generateTanksCameras(poses, width, height, FovX, FovY):
    w2cs = np.linalg.inv(poses)
    cam_infos = []
    for idx, w2c in enumerate(w2cs):
        R = w2c[:3, :3].transpose()
        T = w2c[:3, 3]
        
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=None, dtumask=None, blendermask=None, point3D_ids=None, near_far=None,
                              image_path=None, image_name=None, width=width, height=height)
        cam_infos.append(cam_info)
    
    return cam_infos


def CreateTanksSpiral2(path):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 != 0]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 == 0]
    
    N_sparse = 3
    idx_train = np.linspace(0, len(train_cam_infos) - 1, N_sparse)
    idx_train = [round(i) for i in idx_train]
    train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_train]
    
    train_poses = []
    for cam in train_cam_infos:
        w2c_init = np.zeros((4, 4))
        w2c_init[:3, :3] = cam.R.transpose()
        w2c_init[:3, 3] = cam.T
        w2c_init[3, 3] = 1.0
        train_poses.append(w2c_init)
    train_poses = np.linalg.inv(np.stack(train_poses, axis=0))
    
    virtual_poses = interpolate_virtual_poses_sequential(train_poses, 30)
    virtual_poses = np.concatenate([virtual_poses, virtual_poses[::-1]], axis=0)
    
    render_cam_infos = generateTanksCameras(virtual_poses, train_cam_infos[0].width, train_cam_infos[0].height, train_cam_infos[0].FovX, train_cam_infos[0].FovY)

    nerf_normalization = getNerfppNorm(render_cam_infos)

    
    scene_info = SceneInfo(point_cloud=None,
                           match_data=None,
                           base_cameras = None,
                           train_cameras=None,
                           test_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "DTU": readDTUSceneInfo,
    "Tanks": readTanksSceneInfo,
    "LLFFVideo": CreateLLFFSpiral,
    "TanksVideo": CreateTanksSpiral2,
}