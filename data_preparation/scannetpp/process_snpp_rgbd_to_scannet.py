# Modified from: https://github.com/ayushjain1144/odin/tree/0cd49cb3a52e88869e0a983a1b2f2d6277041b9e/data_preparation
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from PIL import Image
from natsort import natsorted
from glob import glob
import multiprocessing as mp
import json
import torch.nn.functional as F
from tqdm import tqdm

import torch

import ipdb
st = ipdb.set_trace

DATA_PATH = '/data/group_data/katefgroup-ssd/datasets/scannetpp_full/data'
FRAMES_PROCESSED = '/data/group_data/katefgroup/language_grounding/SEMSEG_100k/scannetpp_frames'

# this is width, height following opencv convention
INPUT_IMAGE_SIZE = (1920, 1440)
OUTPUT_IMAGE_SIZE = (640, 480)
FRAME_SKIP = 30

def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def region_to_camera_mapping_fn(pano_file):
    pano_region = open(pano_file).readlines()
    region_to_camera_mapping = {}
    for line in pano_region:
        idx, camera, region, _ = line.split(' ')
        if region in region_to_camera_mapping:
            region_to_camera_mapping[region].append(camera)
        else:
            region_to_camera_mapping[region] = [camera]

    if '-1' in region_to_camera_mapping:
        del region_to_camera_mapping['-1']

    return region_to_camera_mapping


def fix_intrinsics(intrinsics, original_size=(1920, 1440), new_size=(640, 480)):
    intrinsics[0, :] *= new_size[0] / original_size[0]
    intrinsics[1, :] *= new_size[1] / original_size[1]
    return intrinsics


def load_image(path, new_size=(640, 480), mode="bilinear"):
    image = Image.open(path)
    if mode == "nearest":
        image = image.resize(new_size, Image.NEAREST)
    else:
        image = image.resize(new_size, Image.BILINEAR)
    return np.array(image)


def load_and_process_pose_intrinsic(pose_intrinsics_path):
    with open(pose_intrinsics_path, "r") as f:
        rtk_json = json.load(f)
    image_ids = natsorted(list(rtk_json.keys()))
    image_ids = image_ids[::FRAME_SKIP]
    
    for i, image_id in enumerate(image_ids):
        frame = rtk_json[image_id]
        pose = np.array(frame["aligned_pose"])
        intrinsic = np.array(frame["intrinsic"])
        intrinsic = fix_intrinsics(intrinsic, original_size=INPUT_IMAGE_SIZE, new_size=OUTPUT_IMAGE_SIZE)
        intrinsic_ = np.eye(4)
        intrinsic_[:3, :3] = intrinsic
        intrinsic = intrinsic_

        if i == 0:
            poses = []
            intrinsics = []

        poses.append(pose)
        intrinsics.append(intrinsic)
    return poses, intrinsics

def load_raw_data(
        pose_intrinsic_path, depths_path,
        rgbs_path, scene_name,
):
    poses, intrinsics = load_and_process_pose_intrinsic(pose_intrinsic_path)

    depths = [load_image(depth_path, new_size=OUTPUT_IMAGE_SIZE, mode='nearest') for i, depth_path in
               enumerate(natsorted(glob(os.path.join(depths_path, "**")))) if i % FRAME_SKIP == 0]
    rgbs = [load_image(rgb_path, new_size=OUTPUT_IMAGE_SIZE, mode='bilinear') for i, rgb_path in enumerate(natsorted(
        glob(os.path.join(rgbs_path, "**")))) if i % FRAME_SKIP == 0
    ]
    

    OUTPUT_DIR = f"{FRAMES_PROCESSED}/{scene_name}"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # color dir
    COLOR_DIR = f"{OUTPUT_DIR}/color"
    if not os.path.exists(COLOR_DIR):
        os.makedirs(COLOR_DIR)

    # depth dir
    DEPTH_DIR = f"{OUTPUT_DIR}/depth"
    if not os.path.exists(DEPTH_DIR):
        os.makedirs(DEPTH_DIR)

    # pose dir
    POSE_DIR = f"{OUTPUT_DIR}/pose"
    if not os.path.exists(POSE_DIR):
        os.makedirs(POSE_DIR)

    # intrinsic dir
    INTRINSIC_DIR = f"{OUTPUT_DIR}/intrinsic"
    if not os.path.exists(INTRINSIC_DIR):
        os.makedirs(INTRINSIC_DIR)

    for i, (pose, depth, rgb, intrinsic) in enumerate(zip(
            poses, depths, rgbs, intrinsics)):
        index = i * FRAME_SKIP
        # save color image
        rgb_image = Image.fromarray(rgb.astype(np.uint8))
        rgb_image.save(f"{COLOR_DIR}/{index}.jpg")

        # save depth by saving the numpy array
        image = Image.fromarray(depth.astype(np.uint16))
        image.save(f"{DEPTH_DIR}/{index}.png")

        # save pose by writing the pose in a txt file
        file = open(f"{POSE_DIR}/{index}.txt", "w")
        for item in pose:
            for j in range(len(item)):
                file.write(str(item[j]) + " ")
            file.write("\n")
        file.close()

        # save intrinsic by saving the numpy array
        file = open(f"{INTRINSIC_DIR}/{index}.txt", "w")
        for item in intrinsic:
            for j in range(len(item)):
                file.write(str(item[j]) + " ")
            file.write("\n")
        file.close()


def process_scene(scene):
    pose_intrinsic_path = os.path.join(DATA_PATH, scene, 'iphone/pose_intrinsic_imu.json')
    depths_path = os.path.join(DATA_PATH, scene, 'iphone/depth')
    rgb_path = os.path.join(DATA_PATH, scene, 'iphone/rgb')
    
    if not os.path.exists(rgb_path):
        print(f"RGB path does not exist for scene {scene}, skipping...")
        return

    load_raw_data(
        pose_intrinsic_path, depths_path,
        rgb_path,
        scene,
    )


if __name__ == "__main__":
    scene_list = os.listdir(DATA_PATH)

    # remove scenes already processed
    scene_list = [
        scene for scene in scene_list
        if os.path.exists(f"{FRAMES_PROCESSED}/{scene}")
    ]

    print(f"Processing {len(scene_list)} scenes...")

    if len(scene_list) == 0:
        print("Nothing to do. All scenes already processed.")
        exit(0)
        
        
    num_processes = 32
    with mp.Pool(processes=num_processes) as pool:
        with tqdm(
            total=len(scene_list),
            desc="Processing scenes",
            unit="scene",
            dynamic_ncols=True,
        ) as pbar:
            for _ in pool.imap_unordered(process_scene, scene_list):
                pbar.update(1)

