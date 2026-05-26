# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple
from copy import deepcopy
from PIL import Image
import numpy as np
import ipdb
import torch
from torchvision import transforms as pt_transforms
import math

from detectron2.data import transforms as T
from fvcore.transforms.transform import NoOpTransform
from detectron2.structures import Instances
from qwen3d.modeling.backproject.backproject import backprojector_dataloader, backprojector


st = ipdb.set_trace

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def convert_instance_image_to_detectron(instance_image, multiplier=1000):
    """
    Converts an instance image to a detectron2 instance dict
        Input: instance_image: H x W (tensor int64)
        Output: Instances object
    """
    instances = Instances((instance_image.shape[0], instance_image.shape[1]))

    unique_instances = torch.unique(instance_image)
    gt_masks = []
    gt_classes = []
    instance_ids = []
    for inst_id in unique_instances:
        label = inst_id // multiplier
        if label == 0:
            continue
        mask = instance_image == inst_id
        gt_masks.append(mask)
        # 0-index the class labels
        gt_classes.append(label - 1)
        instance_ids.append(inst_id)

    if len(gt_masks) == 0:
        instances.gt_masks = torch.zeros(
            (0, instance_image.shape[0], instance_image.shape[1])
        )
        instances.gt_classes = torch.zeros(0)
        instances.instance_ids = torch.zeros(0)
        return instances

    instances.gt_masks = torch.stack(gt_masks)
    instances.gt_classes = torch.tensor(gt_classes)
    instances.instance_ids = torch.tensor(instance_ids)
    return instances


def augment_depth_intrinsics(
    depth, intrinsic, transforms,
    is_train=False,

):
    depth_transforms = deepcopy(transforms)
    # st()
    if is_train:
        depth_transforms[-1].pad_value = 0.0
        depth_transforms[0].interp = Image.NEAREST
        depth = T.apply_transform_gens(
            depth_transforms, T.AugInput(depth)
        )[0].image

        # get camera intrinsics
        if not isinstance(depth_transforms[0], NoOpTransform):
            intrinsic[0] /= depth_transforms[0].h / depth_transforms[0].new_h
            intrinsic[1] /= depth_transforms[0].w / depth_transforms[0].new_w

        # for crop transform
        assert depth_transforms[1].__class__.__name__ == "CropTransform"
        intrinsic[0, 2] -= depth_transforms[1].x0
        intrinsic[1, 2] -= depth_transforms[1].y0
    else:
        depth_transforms[0].interp = Image.NEAREST
        depth = T.apply_transform_gens(
            depth_transforms, T.AugInput(depth)
        )[0].image
        intrinsic[0] /= depth_transforms[0].h / depth_transforms[0].new_h
        intrinsic[1] /= depth_transforms[0].w / depth_transforms[0].new_w

        if len(depth_transforms) > 1:
            if len(depth_transforms) == 2 and depth_transforms[1].__class__.__name__ == "PadTransform":
                pass
            elif len(depth_transforms) == 2 and depth_transforms[1].__class__.__name__ == "CropTransform":
                intrinsic[0, 2] -= depth_transforms[1].x0
                intrinsic[1, 2] -= depth_transforms[1].y0
            elif len(depth_transforms) == 3 and depth_transforms[1].__class__.__name__ == "CropTransform":
                intrinsic[0, 2] -= depth_transforms[1].x0
                intrinsic[1, 2] -= depth_transforms[1].y0
                assert depth_transforms[2].__class__.__name__ == "PadTransform"
            else:
                raise ValueError("Invalid depth_transforms length")

    return depth, intrinsic


def custom_image_augmentations(is_train):
    if is_train:
        return pt_transforms.Compose(
            [
                pt_transforms.ToPILImage(),
                pt_transforms.RandomApply(
                    [pt_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
            ]
        )
    else:
        return None


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if is_train:
        image_size = cfg.INPUT.IMAGE_SIZE
        min_scale = cfg.INPUT.MIN_SCALE
        max_scale = cfg.INPUT.MAX_SCALE

        augmentation = []

        augmentation.extend(
            [
                T.RandomApply(
                    T.ResizeScale(
                        min_scale=min_scale,
                        max_scale=max_scale,
                        target_height=image_size,
                        target_width=image_size,
                    )
                ),
                T.FixedSizeCrop(crop_size=(image_size, image_size)),
            ]
        )

    else:
        augmentation = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                cfg.INPUT.MIN_SIZE_TEST_SAMPLING,
            ),
        ]

    if cfg.BREAKPOINT:
        print(f"build_transform_gen augmentation: {augmentation}")

    return augmentation


def build_transform_gen_grounding(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if is_train:
        image_size = cfg.INPUT.IMAGE_SIZE
        # min_scale = cfg.INPUT.MIN_SCALE
        # max_scale = cfg.INPUT.MAX_SCALE
        # print(f"image_size: {image_size}, min_scale: {cfg.INPUT.MIN_SIZE_TEST}, max_scale: {cfg.INPUT.MAX_SIZE_TEST}, min_size_test_sampling: {cfg.INPUT.MIN_SIZE_TEST_SAMPLING}")

        augmentation = [
            T.ResizeShortestEdge(
                image_size,
                image_size,
                "choice",
            ),
        ]

    else:
        augmentation = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                cfg.INPUT.MIN_SIZE_TEST_SAMPLING,
            ),
        ]

    return augmentation


def get_multiview_xyz(
    shape: Tuple[int, int, int],
    size_divisibility: int,
    depths: List[torch.Tensor],
    poses: List[torch.Tensor],
    intrinsics: List[torch.Tensor],
    is_train: bool,
    augment_3d: bool,
    interpolation_method: str = "bilinear",
    mask_valid: bool = False,
    mean_center: bool = False,
    do_rot_scale: bool = False,
    scannet_pc: Optional[torch.Tensor] = None,
    align_matrix=None,
    vil3d=False,
    scales=None,
    min_pixel=3136, 
    max_pixel=390000,
):
    # if scales is None:
    #     scales = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
    # else:
    #     assert len(scales) == 2
    #     scales = {f"res{i+2}": s for i, s in enumerate(scales)}
    #     assert scales["res3"] < scales["res2"], f"Expecting scales to be decreasing, got {scales}"

    if scales is None:
        scales = [32, 16, 8, 4]
    else:
        assert len(scales) == 2
        assert scales[0] > scales[1], f"Expecting scales to be decreasing, got {scales}"

    v, h, w = shape
    # pad_h = int(np.ceil(h / size_divisibility) * size_divisibility - h)
    # pad_w = int(np.ceil(w / size_divisibility) * size_divisibility - w)
    # H_padded = h + pad_h
    # W_padded = w + pad_w
    # features = {
    #     k: torch.zeros(v, 1, H_padded // s, W_padded // s) for k, s in scales.items()
    # }
    h_pad = max(scales[0], round_by_factor(h, scales[0]))
    w_pad = max(scales[0], round_by_factor(w, scales[0]))
    if h_pad * w_pad > max_pixel:
        beta = math.sqrt((h * w) / max_pixel)
        h_pad = max(scales[0], floor_by_factor(h / beta, scales[0]))
        w_pad = max(scales[0], floor_by_factor(w / beta, scales[0]))
    elif h_pad * w_pad < min_pixel:
        beta = math.sqrt(min_pixel / (h * w))
        h_pad = ceil_by_factor(h * beta, scales[0])
        w_pad = ceil_by_factor(w * beta, scales[0])
    ds_h = h_pad // scales[0]
    ds_w = w_pad // scales[0]

    features = {"res2": torch.zeros(v, 1, ds_h, ds_w)}

    for i in range(1, len(scales)):
        upsample_factor = scales[0] / scales[i] 
        assert upsample_factor.is_integer(), f"Following resolutions must be divisible by first resolution, got upsample factor {upsample_factor}"
        features[f"res{i+2}"] = torch.zeros(v, 1, ds_h*int(upsample_factor), ds_w*int(upsample_factor))

    depths = torch.stack(depths)
    poses = torch.stack(poses)

    intrinsics = torch.stack(intrinsics)
    augment = is_train and augment_3d

    multi_scale_xyz, scannet_pc, original_xyz = backprojector_dataloader(
        list(features.values()),
        depths,
        poses,
        intrinsics,
        augment=augment,
        method=interpolation_method,
        scannet_pc=scannet_pc,
        padding=None,
        mask_valid=mask_valid,
        mean_center=mean_center,
        do_rot_scale=do_rot_scale,
        align_matrix=align_matrix,
        vil3d=vil3d,
    )
    multi_scale_xyz = multi_scale_xyz[::-1]
    return multi_scale_xyz, scannet_pc, original_xyz[0], ds_h, ds_w


def get_multiview_xyz_cuda(
    depths: torch.Tensor,
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    is_train: bool,
    augment_3d: bool,
    interpolation_method: str = "bilinear",
    mask_valid: bool = False,
    mean_center: bool = False,
    do_rot_scale: bool = False,
    scannet_pc: Optional[torch.Tensor] = None,
    align_matrix=None,
):
    scales = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
    b, v, h, w = depths.shape
    multi_scale_shapes = [
        (b, v, h // s, w // s) for k, s in scales.items()
    ]
    
    augment = is_train and augment_3d

    if scannet_pc is None or all([x is None for x in scannet_pc]) or len(scannet_pc) == 1:
        if type(scannet_pc) is list:
            if scannet_pc[0] is None:
                scannet_pc = None
            else:
                assert len(scannet_pc) == 1
                scannet_pc = scannet_pc[0][None]
        
        if isinstance(align_matrix, list):
            if not all(torch.allclose(x, align_matrix[0]) for x in align_matrix):
                print(f"WARNING: align_matrix is not the same for all views: {align_matrix[0], align_matrix[1]}")
                raise ValueError("align_matrix is not the same for all views")
                
            align_matrix = align_matrix[0]
        multi_scale_xyz, scannet_pc, original_xyz = backprojector(
            multi_scale_shapes,
            depths,
            poses,
            intrinsics,
            augment=augment,
            method=interpolation_method,
            scannet_pc=scannet_pc,
            padding=(0, 0),
            mask_valid=mask_valid,
            mean_center=mean_center,
            do_rot_scale=do_rot_scale,
            align_matrix=align_matrix,
        )
        multi_scale_xyz = multi_scale_xyz[::-1]
    else:
        # we need to loop
        multi_scale_xyz_list = []
        scannet_pc_list = []
        original_xyz_list = []
        for i in range(b):
            multi_scale_xyz_i, scannet_pc_i, original_xyz_i = get_multiview_xyz_cuda(
                depths[i][None],
                poses[i][None],
                intrinsics[i][None],
                is_train,
                augment_3d,
                interpolation_method,
                mask_valid,
                mean_center,
                do_rot_scale,
                scannet_pc[i][None].to(depths.device) if scannet_pc[i] is not None else None,
                align_matrix[i].to(depths.device) if align_matrix is not None else None,
            )
            multi_scale_xyz_list.append(multi_scale_xyz_i)
            scannet_pc_list.append(scannet_pc_i[0] if scannet_pc_i is not None else None)
            original_xyz_list.append(original_xyz_i)

        scannet_pc = scannet_pc_list
        original_xyz = original_xyz_list
        multi_scale_xyz = [torch.cat(x, dim=0) for x in zip(*multi_scale_xyz_list)]

    if type(scannet_pc) is not list and scannet_pc is not None:
        assert len(scannet_pc) == 1
        scannet_pc = [scannet_pc]
    return multi_scale_xyz, scannet_pc, original_xyz


def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error

def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2+dy**2+dz**2)
    return trans_error

def compute_ATE(gt, pred):
    # small: 0.00058, large: 0.01099
    """Compute RMSE of ATE
    Args:
        gt (4x4 array dict): ground-truth poses
        pred (4x4 array dict): predicted poses
    """
    errors = []
    idx_0 = list(pred.keys())[0]
    gt_0 = gt[idx_0]
    pred_0 = pred[idx_0]
    for i in pred:
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3] 

        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]

        align_err = gt_xyz - pred_xyz
        errors.append(np.sqrt(np.sum(align_err ** 2)))
    ate = np.sqrt(np.mean(np.asarray(errors) ** 2)) 
    return ate

def apply_pose_noise(cfg, pose_matrix):
    from scipy.spatial.transform import Rotation as R
    """
    Add Gaussian noise to the translation and rotation components of a 4x4 pose matrix.

    Parameters:
    - pose_matrix: 4x4 numpy array representing the pose.
    - translation_std: Standard deviation of the Gaussian noise for translation (float or array-like of size 3).
    - rotation_std: Standard deviation of the Gaussian noise for rotation (float or array-like of size 3).

    Returns:
    - pose_noisy: 4x4 numpy array of the noisy pose.
    """
    translation_std = cfg.POSE_TRANSLATION_NOISE
    rotation_std = cfg.POSE_ROTATION_NOISE

    # Extract rotation and translation components
    R_orig = pose_matrix[:3, :3]
    t_orig = pose_matrix[:3, 3]

    # Ensure translation_std is an array of size 3
    translation_std = np.asarray(translation_std)
    if translation_std.size == 1:
        translation_std = np.full(3, translation_std)
    elif translation_std.size != 3:
        raise ValueError("translation_std must be a scalar or an array-like of size 3.")

    # Add Gaussian noise to translation
    t_noisy = t_orig + np.random.normal(0, translation_std, size=3)

    rotation_std = np.asarray(rotation_std)
    if rotation_std.size == 1:
        rotation_std = np.full(3, rotation_std)
    elif rotation_std.size != 3:
        raise ValueError("rotation_std must be a scalar or an array-like of size 3.")

    rotation_noise = np.random.normal(0, rotation_std, size=3)
    R_noise = R.from_rotvec(rotation_noise).as_matrix()

    R_noisy = R_noise @ R_orig

    pose_noisy = np.eye(4)
    pose_noisy[:3, :3] = R_noisy
    pose_noisy[:3, 3] = t_noisy

    return pose_noisy