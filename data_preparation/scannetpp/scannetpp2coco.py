# Modified from: https://github.com/ayushjain1144/odin/tree/0cd49cb3a52e88869e0a983a1b2f2d6277041b9e/data_preparation
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np
import json
from tqdm import tqdm
import ipdb
import re

from pycococreatortools import pycococreatortools
import pycocotools.mask as mask_util

from global_dirs import SCANNETPP_DATA_DIR, SCANNETPP_SPLITS, LOCATE3D_SCANNETPP_PATH
from qwen3d.global_vars import NAME_MAP_SCANNETPP

st = ipdb.set_trace

DATA_DIR = SCANNETPP_DATA_DIR
SPLITS = SCANNETPP_SPLITS


INFO = {
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

# NAME_MAP_SCANNETPP = {1: 'wall', 2: 'ceiling', 3: 'floor', 4: 'table', 5: 'door', 6: 'ceiling lamp', 7: 'cabinet', 8: 'blinds', 9: 'curtain', 10: 'chair', 11: 'storage cabinet', 12: 'office chair', 13: 'bookshelf', 14: 'whiteboard', 15: 'window', 16: 'box', 17: 'window frame', 18: 'monitor', 19: 'shelf', 20: 'doorframe', 21: 'pipe', 22: 'heater', 23: 'kitchen cabinet', 24: 'sofa', 25: 'windowsill', 26: 'bed', 27: 'shower wall', 28: 'trash can', 29: 'book', 30: 'plant', 31: 'blanket', 32: 'tv', 33: 'computer tower', 34: 'kitchen counter', 35: 'refrigerator', 36: 'jacket', 37: 'electrical duct', 38: 'sink', 39: 'bag', 40: 'picture', 41: 'pillow', 42: 'towel', 43: 'suitcase', 44: 'backpack', 45: 'crate', 46: 'keyboard', 47: 'rack', 48: 'toilet', 49: 'paper', 50: 'printer', 51: 'poster', 52: 'painting', 53: 'microwave', 54: 'board', 55: 'shoes', 56: 'socket', 57: 'bottle', 58: 'bucket', 59: 'cushion', 60: 'basket', 61: 'shoe rack', 62: 'telephone', 63: 'file folder', 64: 'cloth', 65: 'blind rail', 66: 'laptop', 67: 'plant pot', 68: 'exhaust fan', 69: 'cup', 70: 'coat hanger', 71: 'light switch', 72: 'speaker', 73: 'table lamp', 74: 'air vent', 75: 'clothes hanger', 76: 'kettle', 77: 'smoke detector', 78: 'container', 79: 'power strip', 80: 'slippers', 81: 'paper bag', 82: 'mouse', 83: 'cutting board', 84: 'toilet paper', 85: 'paper towel', 86: 'pot', 87: 'clock', 88: 'pan', 89: 'tap', 90: 'jar', 91: 'soap dispenser', 92: 'binder', 93: 'bowl', 94: 'tissue box', 95: 'whiteboard eraser', 96: 'toilet brush', 97: 'spray bottle', 98: 'headphones', 99: 'stapler', 100: 'marker'}

CATEGORIES = [
    {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in NAME_MAP_SCANNETPP.items() 
]

# CATEGORIES = [
#     {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in NAME_MAP20.items() 
# ]

# CATEGORIES_200 = [
#     {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in SCANNET200_NAME_MAP.items()
# ]

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


def _load_json(self, scene_name):
    scene_path = self.dataset_path / "data" / scene_name
    with open(scene_path / "iphone" / "pose_intrinsic_imu.json", "r") as f:
        rtk_json = json.load(f)
    return rtk_json

def polygons_to_bitmask(polygons, height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)


def convert_scannet_to_coco(path, phase):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "depths": [],
        "poses": [],
        "intrinsics": [],
    }
    
    
    # ldata = json.load(open(LOCATE3D_SCANNETPP_PATH,'r'))
    # frames_used = {ldata_['scene_id']: ldata_['frames_used'] for ldata_ in ldata}

    # get list
    scene_ids = read_txt(SPLITS[phase])
    image_ids = []
    for scene_id in scene_ids:
        for image_id in os.listdir(os.path.join(path, scene_id, 'color')):
            image_ids.append(os.path.join(scene_id, image_id.split('.')[0]))
    print("images number in {}: {}".format(path, len(image_ids)))
    
    # for scene_id in tqdm(scene_ids, desc="Processing scenes"):
    #     for image_id in os.listdir(os.path.join(path, scene_id, 'iphone/rgb')):
    #         # num = int(re.search(r'frame_(\d+)\.jpg', image_id).group(1))
    #         # st()
    #         # if num in frames_used[scene_id]:
    #         image_ids.append(os.path.join(scene_id, 'iphone/rgb', image_id.split('.')[0]))
    print("images number in {}: {}".format(path, len(image_ids)))
    
    coco_image_id = 1

    for index in tqdm(range(len(image_ids))):
        print("{}/{}".format(index, len(image_ids)), end='\r')
        scene_id = image_ids[index].split('/')[0]
        image_id = image_ids[index].split('/')[1]
        # rgb_image_size = (1440,1920)
        # depth_image_size = (192, 256)
        image_size = (640, 480)

        ext = 'jpg'
        image_filename = os.path.join(scene_id, 'color', image_id + f'.{ext}')
        image_info = pycococreatortools.create_image_info(coco_image_id, image_filename, image_size)
        coco_output['images'].append(image_info)

        depth_filename = os.path.join(scene_id, 'depth', image_id + '.png')
        depth_info = pycococreatortools.create_image_info(coco_image_id, depth_filename, image_size)
        coco_output['depths'].append(depth_info)
        
        pose_filename = os.path.join(scene_id, 'pose', image_id + '.txt')
        pose_info = pycococreatortools.create_image_info(
            coco_image_id, pose_filename, image_size)
        coco_output['poses'].append(pose_info)
        
        intrinsic_filename = os.path.join(scene_id, 'intrinsic', image_id + '.txt')
        pose_info = pycococreatortools.create_image_info(
            coco_image_id, intrinsic_filename, image_size)
        coco_output['intrinsics'].append(pose_info)

        coco_image_id += 1

    parent_dir = os.path.dirname(path)
    json.dump(coco_output, open(f'{parent_dir}/scannetpp_{phase}.coco.json','w'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # for phase in ['train', 'val', 'two_scene', 'ten_scene']:
    for phase in ['val',]:
        print("Processing phase: ", phase)
        convert_scannet_to_coco(DATA_DIR, phase)
