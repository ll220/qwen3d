# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import itertools
import logging
import numpy as np
import torch
from prettytable import PrettyTable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from qwen3d.utils.knn import knn_points
import pyviz3d.visualizer as viz
from pathlib import Path
import detectron2.utils.comm as comm
from detectron2.evaluation import DatasetEvaluator
from collections import Counter
import json

from qwen3d.utils.bbox_utils import _set_axis_align_bbox, _get_ious, get_3d_box_scanrefer
from qwen3d.utils.misc import all_gather, is_dist_avail_and_initialized

import ipdb
st = ipdb.set_trace

def box_xyzxyz_to_cxcyczwhd(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    x_c = 0.5 * (x0 + x1)
    y_c = 0.5 * (y0 + y1)
    z_c = 0.5 * (z0 + z1)
    w = x1 - x0
    h = y1 - y0
    d = z1 - z0
    return torch.stack([x_c, y_c, z_c, w, h, d], dim=-1)

def visualize_pc_masks_and_bbox(
    pc, color, gt_pcs,
    pred_pcs, gt_bbox, pred_bbox,
    data_dir=None, sample_name=None, inputs=None,
    gt_anchor_pcs=None, gt_anchor_bboxs=None, sr3d_data=None,
    anchor_pcs=None, anchor_bboxs=None, llm_xyz=None
):
    """
    Input
        pc: N X 3
        color: N X 3 (0-255)
        gt_masks: M X N
        pred_masks: M_pred X N
        gt_bbox: M X 6
        pred_bbox: M_pred X 6 (min-max box) 
    """
    point_size = 25
    v = viz.Visualizer()
    v.add_points("RGB", pc,
                 colors=color,
                 alpha=0.8,
                 visible=True,
                 point_size=25)

    if llm_xyz is not None and len(llm_xyz) > 0:
        # voxel-pooled tokens (5 cm voxel centers) that were fed to the LLM
        v.add_points("LLM input (voxel tokens)", np.asarray(llm_xyz, dtype=np.float32),
                     colors=np.tile(np.array([0, 200, 255])[None], (len(llm_xyz), 1)),
                     alpha=0.9, visible=False, point_size=point_size)

     # add gt masks
    masks_colors = [np.tile(np.array([0, 255, 0])[None], (pc.shape[0], 1)) for pc in gt_pcs]
    v.add_points(
        "Instances (GT)", gt_pcs[0],
        colors=masks_colors[0],
        alpha=0.8,
        visible=False,
        point_size=point_size
    )

    # add pred masks
    dists = knn_points(torch.from_numpy(pc[None]).cuda(), torch.from_numpy(pc[None]).cuda(), K=8)[0][0, :, 1:].mean(1)
    threshold = dists.mean() + 2 * dists.std()
    pc = pc[(dists < threshold).cpu().numpy()]
    masks_colors = [np.tile(np.array([255, 0, 0])[None], (pc.shape[0], 1)) for pc in pred_pcs]
    v.add_points(
        "Instances (PRED)", pred_pcs[0],
        colors=masks_colors[0],
        alpha=0.8,
        visible=False,
        point_size=point_size)

    # add gt boxes
    gt_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(gt_bbox)).numpy()
    v.add_bounding_box(
        'Boxes (GT)',
        position=gt_bbox[..., :3][0],
        size=gt_bbox[..., 3:][0],
        color=np.array([0, 255, 0]),
        alpha=0.8,
        edge_width=0.03
    )

    # add pred boxes
    pred_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(pred_bbox)).numpy()
    v.add_bounding_box(
        'Boxes (Pred)',
        position=pred_bbox[..., :3][0],
        size=pred_bbox[..., 3:][0],
        color=np.array([255, 0, 0]),
        alpha=0.8,
        visible=True,
        edge_width=0.03)

    if gt_anchor_pcs is not None:
        anchor_colors = get_color(len(gt_anchor_pcs))
        for i in range(len(gt_anchor_pcs)):
            _anchor_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(gt_anchor_bboxs[i])).numpy()
            _anchor_name = sr3d_data['anchors_names'][i]
            v.add_points(
                f"A PC {i},{_anchor_name[:5]}", gt_anchor_pcs[i],
                colors=np.array(anchor_colors[i])[None].repeat(gt_anchor_pcs[i].shape[0], axis=0),
                alpha=0.8, visible=False, point_size=point_size)
            v.add_bounding_box(
                f'A BB {i},{_anchor_name[:5]}',
                position=_anchor_bbox[..., :3][0],
                size=_anchor_bbox[..., 3:][0],
                color=np.array([0, 255, 0]),
                alpha=0.8,
                edge_width=0.03,
                visible=False
            )

        v.add_labels(
            'Labels',
            [sr3d_data['text_caption'], sr3d_data['target_name'], sr3d_data['anchors_names']],
            [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            [np.array([255.0, 0.0, 0.0]), np.array([0.0, 255.0, 0.0]), np.array([0.0, 0.0, 255.0])],
            visible=True
        )

    if anchor_pcs is not None:
        anchor_colors = get_color(len(anchor_pcs))
        for i in range(0, min(len(gt_anchor_pcs), len(anchor_pcs))):
            _anchor_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(anchor_bboxs[i])).numpy()
            v.add_points(
                f"PA PC {i}", anchor_pcs[i],
                colors=np.array(anchor_colors[i])[None].repeat(anchor_pcs[i].shape[0], axis=0),
                alpha=0.8, visible=False, point_size=point_size)
            v.add_bounding_box(
                f'PA BB {i}',
                position=_anchor_bbox[..., :3][0],
                size=_anchor_bbox[..., 3:][0],
                color=np.array(anchor_colors[i]),
                alpha=0.8,
                edge_width=0.03,
                visible=False
            )

    if gt_anchor_pcs is not None and anchor_pcs is not None:
        print(f"Found {len(gt_anchor_pcs)} GT Anchors and {len(anchor_pcs)} Pred Anchors")

    if data_dir is None:
        data_dir = os.environ['OUTPUT_DIR_PREFIX'] + '/debug/bdetr2_visualizations'

    data_dir = Path(f"{data_dir}/{inputs[0]['dataset_name']}/{sample_name.replace(' ', '_')[:100]}")
    if not data_dir.exists():
        data_dir.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saved to {data_dir}")
    v.save(str(data_dir))

def save_query_png(path, pc, color, gt_pc, pred_pc, gt_bbox, pred_bbox, title):
    """Static view of one query: top-down (full + zoomed) and side view, GT green / prediction red."""
    from matplotlib.patches import Rectangle
    rng = np.random.default_rng(0)
    col = np.asarray(color, dtype=np.float32)
    col = np.clip(col / 255.0 if col.max() > 1.5 else col, 0, 1)
    zmin, zmax = float(pc[:, 2].min()), float(pc[:, 2].max())
    boxes = [b for b in (gt_bbox, pred_bbox) if np.all(np.isfinite(b))]
    zoom = [gt_bbox] if np.all(np.isfinite(gt_bbox)) else boxes  # zoom on the target, not on a huge prediction
    lo = np.min([b[:3] for b in zoom], axis=0) - 1.5
    hi = np.max([b[3:] for b in zoom], axis=0) + 1.5
    cut_z = max(zmin + 0.7 * (zmax - zmin), max(b[5] for b in boxes) + 0.05)  # hide ceiling in top-down views

    def draw(ax, a, b, sel, x_lim=None, y_lim=None):
        if len(sel) > 200000:
            sel = rng.choice(sel, 200000, replace=False)
        ax.set_facecolor("#2b2b2b")
        ax.scatter(pc[sel, a], pc[sel, b], c=col[sel], s=1.5, linewidths=0)
        for pts, c in ((gt_pc, "lime"), (pred_pc, "red")):
            if len(pts):
                ax.scatter(pts[:, a], pts[:, b], c=c, s=3, alpha=0.6, linewidths=0)
        for bx, c in ((gt_bbox, "lime"), (pred_bbox, "red")):
            if np.all(np.isfinite(bx)):
                ax.add_patch(Rectangle((bx[a], bx[b]), bx[3 + a] - bx[a], bx[3 + b] - bx[b],
                                       fill=False, ec=c, lw=2.0))
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if x_lim is not None:
            ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)

    inx = np.all((pc[:, :3] >= lo) & (pc[:, :3] <= hi), axis=1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.2))
    draw(axs[0], 0, 1, np.nonzero(pc[:, 2] <= cut_z)[0]); axs[0].set_title("top-down (ceiling hidden)")
    draw(axs[1], 0, 1, np.nonzero(inx & (pc[:, 2] <= cut_z))[0], (lo[0], hi[0]), (lo[1], hi[1])); axs[1].set_title("top-down zoom (on target)")
    draw(axs[2], 0, 2, np.nonzero(inx)[0], (lo[0], hi[0]), (lo[2], hi[2])); axs[2].set_title("side view (x-z), zoom on target")
    fig.suptitle(title + "\n(green = ground truth, red = prediction)", fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def get_color(max_value: int, colormap='spring'):
    colormap = plt.get_cmap('spring')  # Pink is 0, Yellow is 1
    colors = [mcolors.to_rgb(colormap(i / max_value)) for i in range(max_value)]  # Generate colors
    return (np.array(colors) * 255).astype(int).tolist()

def convert_bbox_to_corners_with_colors(bboxes):
    """
    Convert bounding boxes to a format with 8 corners and deterministically generate a color for each box.

    Args:
    - bboxes (np.array): An array of shape [N, 6] containing bounding boxes.

    Returns:
    - np.array: An array of dictionaries, each with 'corners' and 'color' keys.
    """
    converted = []
    colors = get_color(len(bboxes))
    for idx, bbox in enumerate(bboxes):
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        corners = np.array([
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
        ])
        converted.append({"corners": corners.tolist(), "color": colors[idx]})
    return np.array(converted)

def crop_pcd_to_combined_bbox(pcd: torch.Tensor, bboxes: torch.Tensor, extra_extent: float) -> torch.Tensor:
    """
    Crop point cloud data (PCD) to match the combined extent of all bounding boxes.

    Parameters:
    - pcd: torch.Tensor of shape [M, 6] representing the point cloud data with [xyz rgb].
    - bboxes: torch.Tensor of shape [N, 6] representing the bounding boxes with [xmin, ymin, zmin, xmax, ymax, zmax].

    Returns:
    - torch.Tensor: Cropped PCD matching the combined extent of all bounding boxes.
    """
    # Calculate the min/max extent of all bounding boxes
    combined_min = torch.min(bboxes[:, :3], dim=0).values - extra_extent
    combined_max = torch.max(bboxes[:, 3:], dim=0).values + extra_extent

    # Check if points are within the combined bbox extents
    in_combined_bbox = (pcd[:, 0] >= combined_min[0]) & (pcd[:, 0] <= combined_max[0]) & \
                        (pcd[:, 1] >= combined_min[1]) & (pcd[:, 1] <= combined_max[1]) & \
                        (pcd[:, 2] >= combined_min[2]) & (pcd[:, 2] <= combined_max[2])

    return pcd[in_combined_bbox]

def sample_k_rows(tensor, K):
    indices = torch.randperm(tensor.size(0))[:K]
    return tensor[indices]

class ReferrentialGroundingEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        thresholds,
        topks,
        cfg=None
    ):
        self._logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.thresholds = thresholds
        self.topks = topks
        self.cfg = cfg
        self._cpu_device = torch.device("cpu")
        self.num_viz = 0
        self.total_invalid = 0

    def reset(self):
        self.detection_results = []
        self.mask_detection_results = []
        self.num_viz = 0

    def process(self, inputs, outputs):
        if type(outputs[0]) != dict:
            outputs = outputs[1]
        assert len(inputs) == 1
        for j in range(len(outputs)):
            inputs_ = copy.copy(inputs[0])
            inputs_['sr3d_data'] = [inputs_['sr3d_data'][j]]
            if self.cfg.USE_GT_MASKS:
                self.process_single_gt([inputs_], [outputs[j]])
            else:
                self.process_single([inputs_], [outputs[j]])
                
    def process_single_gt(self, inputs, outputs):
        """
        Process a single input-output pair for 3D referential grounding evaluation.

        This function handles one sample (batch size = 1) from a 3D scene and its corresponding model
        predictions. It extracts predicted instance scores and masks, computes axis-aligned bounding boxes
        (AABBs) for the top-K predictions, and evaluates these predictions against the ground truth target
        (specified in the input). Depending on configuration flags, it may also perform subsampling of the
        point cloud or visualize the results.

        Parameters
        ----------
        inputs : list[dict]
            A list containing one dictionary with the following required keys:
            - "scannet_coords": torch.Tensor of shape (P, 3)
                    3D coordinates of the scene's point cloud (e.g., P ≈ 2.5e5 points).
            - "scannet_color": torch.Tensor of shape (P, 3)
                    Color information for each point (RGB values in 0-255).
            - "scannet_labels": torch.Tensor of shape (P, 2)
                    Per-point labels; the second column (index 1) is used to match the ground truth target.
            - "image_id": str
                    A unique identifier for the scene.
            - "file_name": str
                    File path string used to extract the scene name for visualization.
            - 'dataset_name': str
                    Name of the dataset (used for visualization path).
            - "sr3d_data": list of dict
                    A list with one dictionary that must include:
                    - "target_id": int
                        The ground truth object identifier to detect.
                    - "annotation_id": int or None
                        The annotation id for the target.
                    - "anchor_ids": list of int
                        A list of additional anchor object identifiers (used if visualization is enabled).
                    - "text_caption": str
                        A caption or description for the scene/object.
                    - 'anchors_names': list[str]
                        Names of anchor objects (used for extra viz).
                    - 'target_name': str
                        Name of the target object (used for viz).
                    (Other entries in sr3d_data may be present but are not used in this function.)

        outputs : list[dict]
            A list containing one dictionary with a key "instances_3d", which is a dict that must include:
            - "pred_scores": torch.Tensor of shape (N, C)
                    Prediction confidence scores for N candidate instances. The first column (index 0) is used
                    as the main score (e.g., shape (100, 3)).
            - "pred_masks": torch.Tensor of shape (N, M)
                    Raw prediction masks (pre-thresholding) for each instance over M points (e.g., M ≈ 2.5e5).
            - "scannet_idxs": torch.Tensor or None (optional)
                    Indices used for subsampling the point cloud when FORCE_SUBSAMPLE is enabled.
            - 'reduced_scene_id_to_original_id': torch.Tensor | None
            - (Other keys like "scannet_gt_masks", etc. may be present but are not used.)

        Processing Details
        ------------------
        - The function extracts the predicted scores and applies a sigmoid to the predicted masks,
        thresholding them at 0.5 to yield binary masks.
        - It selects the top-K predictions (with K equal to max(self.topks)) based on the confidence scores.
        - For each of the top predictions, it uses the binary mask to index into "scannet_coords" (a tensor of shape (P, 3))
        to extract the corresponding 3D points, and computes an axis-aligned bounding box via _set_axis_align_bbox.
        If no points are selected, a box with -∞ values is used.
        - If the FORCE_SUBSAMPLE flag is enabled, the function subsamples "scannet_coords", "scannet_labels", and
        "scannet_color" using indices from "scannet_idxs".
        - The ground truth points for the target object (specified by "target_id" in sr3d_data) are
        extracted from "scannet_coords" using "scannet_labels". An axis-aligned bounding box for the ground truth is
        computed, and Intersection-over-Union (IoU) is calculated between each predicted box and the ground truth box.
        - Similarly, mask IoUs are computed between the predicted binary masks and a full ground truth mask derived
        from "scannet_labels".
        - The detection and mask detection results (each a boolean array indicating success at various thresholds)
        are appended to the evaluator's internal lists for later aggregation.

        Returns
        -------
        None
        """

        assert len(inputs) == len(outputs) == 1
        
        # Root word is the first object
        scores = outputs[0]['instances_3d']['pred_scores'][:, 0]
        
        # Average confidence of each prediction mask.
        top_k_weighted_scores = scores
        max_k = max(self.topks)
        
        top_id = torch.argsort(top_k_weighted_scores, descending=True)[:max_k]
        pred_ids = (outputs[0]['instances_3d']['pred_masks'])[top_id, :].argmax(-1)
        reduced_scene_id_to_original_id = outputs[0]['instances_3d']['reduced_scene_id_to_original_id']
        pred_ids = reduced_scene_id_to_original_id[pred_ids]
        target_id = inputs[0]['sr3d_data'][0]['target_id']
        detected = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
        correct = (pred_ids == target_id).cpu().numpy()
        for i in range(len(self.topks)):
            detected[i, :] = np.any(correct[:self.topks[i], None], axis=0)
        self.detection_results.append(detected)
        self.mask_detection_results.append(detected)

    def process_single(self, inputs, outputs):
        assert len(inputs) == len(outputs) == 1
        
        # Root word is the first object
        scores = outputs[0]['instances_3d']['pred_scores'][:, 0]
        pred_masks = F.sigmoid(outputs[0]['instances_3d']['pred_masks'])
        masks = pred_masks > 0.5
        
        # Average confidence of each prediction mask.
        top_k_weighted_scores = scores
        top_k_pred_masks = masks.flatten(1)
        max_k = max(self.topks)
        top_id = torch.argsort(top_k_weighted_scores, descending=True)[:max_k]
        top_masks = top_k_pred_masks[top_id, :].to(self._cpu_device).numpy()

        # locate3d scannetpp always evaluate on subsampled pc
        if self.cfg.FORCE_SUBSAMPLE or 'locate3d_ref_scannetpp' in self.dataset_name:
            dists_close = outputs[0]['instances_3d']['scannet_idxs'].cpu()
            inputs[0]['scannet_coords'] = inputs[0]['scannet_coords'][dists_close]
            inputs[0]['scannet_labels'] = inputs[0]['scannet_labels'][dists_close]
            inputs[0]['scannet_color'] = inputs[0]['scannet_color'][dists_close]

        top_bboxs = []
        pred_pcs = []
        for i in range(max_k):
            if inputs[0]['scannet_coords'].shape[0] == 0:
                return None
                pred_pcs.append(np.zeros((0, 3)))
                top_bboxs.append(np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]))
                continue
            cur_pc = inputs[0]['scannet_coords'].to(self._cpu_device).numpy()[top_masks[i]]
            pred_pcs.append(cur_pc)
            if cur_pc.shape[0] == 0:
                top_bboxs.append(np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]))
            else:
                top_bboxs.append(_set_axis_align_bbox(cur_pc))
        top_bboxs = np.array(top_bboxs)

        if self.cfg.TEST_DATASET_INFERENCE:
            assert len(inputs[0]['sr3d_data'])
            if pred_pcs[0].shape[0] > 0:
                max_ = np.max(pred_pcs[0], axis=0)
                min_ = np.min(pred_pcs[0], axis=0)
            else:
                max_ = np.array([0.0, 0.0, 0.0])
                min_ = np.array([0.0, 0.0, 0.0])

            center = (max_ + min_) / 2.0
            box_size = max_ - min_
            scanrefer_box = get_3d_box_scanrefer(box_size, 0, center)
            self.detection_results.append({
                "scene_id": inputs[0]['image_id'],
                "object_id": inputs[0]['sr3d_data'][0]['target_id'],
                "ann_id": inputs[0]['sr3d_data'][0]['annotation_id'],
                "bbox": scanrefer_box.tolist(),
            })
            return

        target_id = inputs[0]['sr3d_data'][0]['target_id']

        try:
            gt_indices = torch.nonzero(inputs[0]['scannet_labels'][:, 1].cpu() == target_id, as_tuple=True)[0]
            gt_pc = inputs[0]['scannet_coords'][gt_indices, :].cpu().numpy()
            if gt_pc.shape[0] == 0:
                return None
                print("INVALID GT TARGET, no points found")
                self.total_invalid += 1
                gt_bbox = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])[None]
            else:
                gt_bbox = np.expand_dims(_set_axis_align_bbox(gt_pc), axis=0)

            ious = _get_ious(top_bboxs, gt_bbox)
            above_threshold = ious > self.thresholds
            detected = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
            for i in range(len(self.topks)):
                detected[i, :] = np.any(above_threshold[:self.topks[i], :], axis=0)
                
            full_gt_mask = (inputs[0]['scannet_labels'][:, 1].cpu().numpy() == target_id)
            mask_ious = []
            for i in range(max_k):
                pred_mask = top_masks[i].astype(bool)
                intersection = np.logical_and(pred_mask, full_gt_mask).sum()
                union = np.logical_or(pred_mask, full_gt_mask).sum()
                mask_iou = intersection / union if union > 0 else 0.0
                mask_ious.append(mask_iou)
            mask_ious = np.array(mask_ious)[:, None]
            mask_detected = mask_ious > self.thresholds
            mask_result = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
            for i in range(len(self.topks)):
                mask_result[i, :] = np.any(mask_detected[:self.topks[i], :], axis=0)

        except Exception as e:
            if self.cfg.VIZ_EXTRA_REF or self.cfg.TEST_DATASET_INFERENCE:
                print(f"Skipping visualization for {e}. GT Target not found")
                detected = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
                mask_result = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
                gt_bbox = np.array([0, 0, 0, 0, 0, 0])
                gt_pc = np.array([[0, 0, 0]])
            else:
                return

        # Optional subsampled visualization: VIZ_STRIDE=k shows every k-th query. PNGs go to VIZ_PNG_DIR,
        # interactive PyViz3D scenes (VISUALIZE_REF) are limited to the same queries.
        _stride = int(os.environ.get("VIZ_STRIDE", "0"))
        self._q_seen = getattr(self, "_q_seen", 0)
        _do_viz = _stride > 0 and self._q_seen % _stride == 0
        self._q_seen += 1
        if _do_viz and os.environ.get("VIZ_PNG_DIR"):
            try:
                _d = inputs[0]['sr3d_data'][0]
                _scene = inputs[0]['file_name'].split('/')[-3]
                _iou_b = float(np.ravel(locals().get('ious', [[0.0]])[0])[0])
                _iou_m = float(np.ravel(locals().get('mask_ious', [[0.0]])[0])[0])
                _png = f"{os.environ['VIZ_PNG_DIR']}/{self._q_seen - 1:03d}_{_scene}.png"
                save_query_png(
                    _png, inputs[0]['scannet_coords'].numpy(), inputs[0]['scannet_color'].numpy(),
                    gt_pc, pred_pcs[0], np.asarray(gt_bbox).reshape(-1)[:6], np.asarray(top_bboxs[0]),
                    f"{_scene} | \"{_d['text_caption']}\" | target: {_d.get('target_name', '')} | "
                    f"box IoU {_iou_b:.2f}  mask IoU {_iou_m:.2f}")
                np.savez_compressed(
                    _png.replace('.png', '.npz'), pc=inputs[0]['scannet_coords'].numpy().astype(np.float32),
                    color=inputs[0]['scannet_color'].numpy().astype(np.uint8), gt_pc=gt_pc.astype(np.float32),
                    pred_pc=pred_pcs[0].astype(np.float32), gt_bbox=np.asarray(gt_bbox).reshape(-1)[:6],
                    pred_bbox=np.asarray(top_bboxs[0]), caption=_d['text_caption'],
                    target=_d.get('target_name', ''), scene=_scene, box_iou=_iou_b, mask_iou=_iou_m,
                    llm_xyz=outputs[0]['llm_voxel_xyz'].numpy() if 'llm_voxel_xyz' in outputs[0] else np.zeros((0, 3), np.float32))
                with open(f"{os.environ['VIZ_PNG_DIR']}/index.jsonl", "a") as _f:
                    _f.write(json.dumps({"file": os.path.basename(_png), "scene": _scene,
                                         "caption": _d['text_caption'], "target": _d.get('target_name', ''),
                                         "box_iou": _iou_b, "mask_iou": _iou_m}) + "\n")
            except Exception as e:
                print(f"PNG visualization failed: {e}")

        if self.cfg.VISUALIZE_REF and (_stride == 0 or _do_viz):
            gt_anchor_bboxs = None
            gt_anchor_pcs = None
            anchor_pcs = None
            anchor_bboxs = None
            if self.cfg.VIZ_EXTRA_REF:
                anchor_ids = inputs[0]['sr3d_data'][0]['anchor_ids']
                gt_anchor_pcs = []
                gt_anchor_bboxs = []
                for anchor_id in anchor_ids:
                    gt_anchor_pc = inputs[0]['scannet_coords'][inputs[0]['scannet_labels'][:, 1] == anchor_id, :].cpu().numpy()
                    if len(gt_anchor_pc) > 0:
                        gt_anchor_pcs.append(gt_anchor_pc)
                        gt_anchor_bboxs.append(np.expand_dims(_set_axis_align_bbox(gt_anchor_pc), axis=0))
                    else:
                        print(f"Anchor {anchor_id} has no points")

                anchor_pcs = []
                anchor_bboxs = []
                for n in range(1, outputs[0]['instances_3d']['pred_scores'].shape[1]):
                    anchor_scores = outputs[0]['instances_3d']['pred_scores'][:, n]
                    anchor_masks = F.sigmoid(outputs[0]['instances_3d']['pred_masks']) > 0.5
                    top_anchor_id = torch.argmax(anchor_scores).to(anchor_masks.device)
                    anchor_pc = inputs[0]['scannet_coords'][anchor_masks[top_anchor_id].to(inputs[0]['scannet_coords'].device)].cpu().numpy()
                    if anchor_pc.shape[0] > 0:
                        anchor_pcs.append(anchor_pc)
                        anchor_bboxs.append(np.expand_dims(_set_axis_align_bbox(anchor_pc), axis=0))

            scene_name = inputs[0]['file_name'].split('/')[-3] + " " + inputs[0]['sr3d_data'][0]['text_caption']
            visualize_pc_masks_and_bbox(
                pc=inputs[0]['scannet_coords'].numpy(),
                color=inputs[0]['scannet_color'].numpy(),
                gt_pcs=[gt_pc],
                pred_pcs=pred_pcs,
                gt_bbox=gt_bbox,
                pred_bbox=top_bboxs[0][None],
                data_dir=self.cfg.VISUALIZE_LOG_DIR,
                sample_name=scene_name,
                inputs=inputs,
                gt_anchor_pcs=gt_anchor_pcs,
                gt_anchor_bboxs=gt_anchor_bboxs,
                sr3d_data=inputs[0]['sr3d_data'][0],
                anchor_pcs=anchor_pcs,
                anchor_bboxs=anchor_bboxs,
                llm_xyz=outputs[0]['llm_voxel_xyz'].numpy() if 'llm_voxel_xyz' in outputs[0] else None
            )

        self.detection_results.append(detected)
        self.mask_detection_results.append(mask_result)

    def evaluate(self):
        if is_dist_avail_and_initialized():
            detection_results = all_gather(self.detection_results)
            detection_results = list(itertools.chain(*detection_results))
            mask_detection_results = all_gather(self.mask_detection_results)
            mask_detection_results = list(itertools.chain(*mask_detection_results))
            if not comm.is_main_process():
                return {}
        else:
            print("TOTAL INVALID GT TARGETS:", self.total_invalid)
            detection_results = self.detection_results
            mask_detection_results = self.mask_detection_results

        if self.cfg.TEST_DATASET_INFERENCE:
            try:
                Path(self.cfg.TEST_RESULT_EXPORT_PATH).mkdir(parents=True, exist_ok=True)
                print(f'exporting test results to {self.cfg.TEST_RESULT_EXPORT_PATH}/{self.dataset_name}_test_results.json')
                with open(f'{self.cfg.TEST_RESULT_EXPORT_PATH}/{self.dataset_name}_test_results.json', 'w') as json_file:
                    json.dump(detection_results, json_file, indent=4)
            except Exception as e:
                print(f"Error exporting test results: {e}")
                st()
            return None

        self.detection_results = []
        self.mask_detection_results = []
        detection_results = np.array(detection_results).astype(np.float32).mean(axis=0)
        mask_detection_results = np.array(mask_detection_results).astype(np.float32).mean(axis=0)
            
        table = PrettyTable()
        table.field_names = ['Dataset', ''] + ['thresold = ' + str(thres) for thres in self.thresholds]
        for i in range(len(self.topks)):
            row = [self.dataset_name, f'top {self.topks[i]} scores (BOX)'] + ['{:.3f}'.format(detection_results[i, j]) for j in range(len(self.thresholds))]
            table.add_row(row)
        for i in range(len(self.topks)):
            row = [self.dataset_name, f'top {self.topks[i]} scores (MASK)'] + ['{:.3f}'.format(mask_detection_results[i, j]) for j in range(len(self.thresholds))]
            table.add_row(row)
        print(table)

        res = {}
        
        for i in range(len(self.topks)):
            for j in range(len(self.thresholds)):
                res[f"top_{self.topks[i]}_threshold_{self.thresholds[j]}"] = detection_results[i, j]

        for i in range(len(self.topks)):
            for j in range(len(self.thresholds)):
                res[f"mask_top_{self.topks[i]}_threshold_{self.thresholds[j]}"] = mask_detection_results[i, j]
        return res
