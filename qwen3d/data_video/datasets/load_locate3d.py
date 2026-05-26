# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Optional
import pandas as pd
import os
import ast
import json
import copy
from pathlib import Path
from detectron2.data import DatasetCatalog, MetadataCatalog
from qwen3d.global_vars import NAME_MAP_SCANNETPP

import ipdb
st = ipdb.set_trace


# /data/group_data/katefgroup-ssd/datasets/locate_3d
_PREDEFINED_SPLITS_REF_LOCATE3D = {
    'locate3d_ref_scannetpp_val_single': ('val_scannetpp.json'),
}


def load_ref(
        csv_file: str,
        return_scene_with_batched_captions: bool = False,
        return_scene_batch_size: int = 2,
        subsample_scenes: Optional[int] = None,
    ):
    with open(csv_file, 'r') as f:
        sr3d_data = json.load(f)
        print(f"Loaded {len(sr3d_data)} scenes from {csv_file}. JSON mode.")

    scannet_split = 'scannetpp_val_single'
    # if 'train' in csv_file or 'debug' in csv_file:
    #     scannet_split = 'scannet200_context_instance_train_200cls_single_highres_100k'
    # elif 'test' in csv_file and 'scanrefer' in csv_file.lower():
    #     scannet_split = 'scannet200_context_instance_test_200cls_single_highres_100k'
    # else:
    #     scannet_split = 'scannet200_context_instance_val_200cls_single_highres_100k'

    scannet_scenes = DatasetCatalog.get(scannet_split)
    if subsample_scenes is not None:
        import random
        random.seed(0)
        scannet_scenes = random.sample(scannet_scenes, subsample_scenes)

    scene_name_to_list_id = {}
    for i in range(len(scannet_scenes)):
        scene_name_to_list_id[scannet_scenes[i]['image_id']] = i

    if return_scene_with_batched_captions:
        tmp_sr3d_data = defaultdict(list)
        cleared_lists = []
        for sr3d_instance in sr3d_data:
            if "scan_id" not in sr3d_instance:
                sr3d_instance["scan_id"] = sr3d_instance.pop("scene_id")
            if sr3d_instance['scan_id'] not in scene_name_to_list_id:
                # assert False
                continue
            if len(tmp_sr3d_data[sr3d_instance['scan_id']]) >= return_scene_batch_size:
                # We fill up the entry in the dict until it reaches our desired batch size
                cleared_lists.append(tmp_sr3d_data[sr3d_instance['scan_id']])
                tmp_sr3d_data[sr3d_instance['scan_id']] = []

            tmp_sr3d_data[sr3d_instance['scan_id']].append(sr3d_instance)
        sr3d_data = list(tmp_sr3d_data.values()) + cleared_lists
    # import pdb; pdb.set_trace()
    
    return sr3d_data, scannet_scenes, scene_name_to_list_id


def get_snpp_meta(name_map):
    dataset_categories = [
        {'id': key, 'name': item, 'supercategory': 'nyu40'} for key, item in name_map.items()
    ]

    thing_ids = [k["id"] for k in dataset_categories]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in dataset_categories]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_ref_locate3d(root):
    return_scene_batch_size = int(os.getenv('RETURN_SCENE_BATCH_SIZE', 6))
    assert 'scannetpp_val_single' in DatasetCatalog
    assert 'scannetpp_val_single' in DatasetCatalog
    
    for key, csv_file in _PREDEFINED_SPLITS_REF_LOCATE3D.items():
        DatasetCatalog.register(key, lambda csv_file=csv_file: load_ref(os.path.join(root, csv_file)))
        MetadataCatalog.get(key).set(**get_snpp_meta(NAME_MAP_SCANNETPP))

        DatasetCatalog.register(f"{key}_batched", lambda csv_file=csv_file: load_ref(
            os.path.join(root, csv_file), return_scene_with_batched_captions=True, return_scene_batch_size=return_scene_batch_size))

        if 'val' in key:
            name = key.replace('val', 'val_50')
            DatasetCatalog.register(f"{name}_batched", lambda csv_file=csv_file: load_ref(
                os.path.join(root, csv_file),
                return_scene_with_batched_captions=True,
                return_scene_batch_size=return_scene_batch_size,
                subsample_scenes=50
            ))

            name = key.replace('val', 'val_5')
            DatasetCatalog.register(f"{name}_batched", lambda csv_file=csv_file: load_ref(
                os.path.join(root, csv_file),
                return_scene_with_batched_captions=True,
                return_scene_batch_size=return_scene_batch_size,
                subsample_scenes=5
            ))

        if 'test' in key:
            name = key.replace('test', 'test_2')
            DatasetCatalog.register(f"{name}_batched", lambda csv_file=csv_file: load_ref(
                os.path.join(root, csv_file),
                return_scene_with_batched_captions=True,
                return_scene_batch_size=return_scene_batch_size,
                subsample_scenes=5
            ))
