# Modified from: https://github.com/ayushjain1144/odin/tree/0cd49cb3a52e88869e0a983a1b2f2d6277041b9e/data_preparation
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# TODO: Make path generic.
DATA_DIR = '/path/to/SEMSEG_100k/frames_square_highres' # This is path to the folder which contains the RGRB-D data (not scannet point cloud data)
SPLITS_PATH = 'splits/scannet_splits'

SPLITS = {
    'train':   f'{SPLITS_PATH}/scannetv2_train.txt',
    'val':     f'{SPLITS_PATH}/scannetv2_val.txt',
    'trainval': f'{SPLITS_PATH}/scannetv2_trainval.txt',
    'test':    f'{SPLITS_PATH}/scannetv2_test.txt',
    'two_scene': f'{SPLITS_PATH}/two_scene.txt',
    'ten_scene': f'{SPLITS_PATH}/ten_scene.txt'
}


SCANNETPP_DATA_DIR = '/data/group_data/katefgroup/language_grounding/scannetpp/data'
SCANNETPP_SPLITS_PATH = '/data/group_data/katefgroup/language_grounding/scannetpp/splits'

SCANNETPP_SPLITS = {
    'one_scene': f'{SCANNETPP_SPLITS_PATH}/one_scene.txt',
    'debug': f'{SCANNETPP_SPLITS_PATH}/one_scene.txt',
    'nvs_sem_train': f'{SCANNETPP_SPLITS_PATH}/nvs_sem_train.txt',
    'val': f'{SCANNETPP_SPLITS_PATH}/nvs_sem_val.txt',  
    'nvs_test': f'{SCANNETPP_SPLITS_PATH}/nvs_test.txt',
    'nvs_test_small': f'{SCANNETPP_SPLITS_PATH}/nvs_test_small.txt',
    'sem_test': f'{SCANNETPP_SPLITS_PATH}/sem_test.txt',
}
LOCATE3D_SCANNETPP_PATH = '/home/ayushj2/lucy_qwen3d/nav25/val_scannetpp.json'