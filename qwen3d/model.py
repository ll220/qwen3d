# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import random
from pathlib import Path
from typing import Tuple

import detectron2.utils.comm as comm
import numpy as np
import torch
import wandb
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from transformers import AutoProcessor

from feature_map_tools.qwen3d_encoder import Qwen3DEncoder
from qwen3d.conversation_template import (
    TEXT_INPUT_GEN,
    TEXT_INPUT_GEN_2D,
    TEXT_INPUT_GEN_3D,
    TEXT_INPUT_TRAIN,
    TEXT_INPUT_TRAIN_2D,
    TEXT_INPUT_TRAIN_3D,
)
from qwen3d.data_video.sentence_utils import (
    convert_grounding_to_od_logits,
    convert_grounding_to_od_logits_ref,
)
from qwen3d.global_vars import (
    LEARNING_MAP_INV,
    MATTERPORT_ALL_CLASSES_TO_21,
    SCANNET200_LEARNING_MAP_INV,
)
from qwen3d.model_data import load_scannet_data, prepare_targets, slice_tensor
from qwen3d.modeling.backproject.backproject import (
    interpolate_feats_3d,
    multiscsale_voxelize,
    voxel_map_to_source,
)
from qwen3d.modeling_qwen2_5_vl_modified import (
    Qwen2_5_VLForConditionalGeneration,
    build_full_mask,
)
from qwen3d.utils import vis_utils
from qwen3d.utils.misc import is_dist_avail_and_initialized
from qwen3d.utils.util_3d import sample_2D_indices

from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher
from .utils.memory import retry_if_cuda_oom

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class Qwen3D(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        mask_decoder: nn.Module,
        criterion: VideoSetCriterion,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        decoder_3d,
        supervise_sparse,
        eval_sparse,
        cfg,
    ):
        """
        Args:
            mask_decoder: transformer mask decoder that produces per-query masks/logits.
            criterion: module that defines the loss.
            num_queries: number of object queries.
            object_mask_threshold: threshold to filter queries by classification score.
            overlap_threshold: overlap threshold used in inference.
            metadata: dataset metadata (class names, etc.).
            size_divisibility: input H/W must be divisible by this value.
            sem_seg_postprocess_before_inference: whether to resize predictions back to
                original input size before semantic segmentation inference.
            pixel_mean, pixel_std: per-channel mean/std for image normalization.
            decoder_3d: whether the 3D decoder is enabled.
            supervise_sparse: whether to supervise on sparse 3D points.
            eval_sparse: whether to evaluate on sparse 3D points.
            cfg: full detectron2 config node.
        """
        super().__init__()
        self.this_device = torch.cuda.current_device()

        if cfg.USE_WANDB:
            if not is_dist_avail_and_initialized() or comm.is_main_process():
                name = (
                    cfg.OUTPUT_DIR.split("/")[-1]
                    if cfg.WANDB_NAME is None
                    else cfg.WANDB_NAME
                )
                wandb.init(
                    entity=cfg.WANDB_ENTITY,
                    project=cfg.WANDB_PROJECT,
                    sync_tensorboard=True,
                    name=name,
                    resume="allow",
                    config=cfg,
                    mode="online",
                    settings=wandb.Settings(init_timeout=120),
                    id=name,
                )

        self.qwen_processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            min_pixels=cfg.INPUT.MIN_PIXEL,
            max_pixels=cfg.INPUT.MAX_PIXEL,
        )
        self.qwen_processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|pointcloud_pad|>"]}
        )
        pointcloud_token_id = self.qwen_processor.tokenizer.convert_tokens_to_ids(
            "<|pointcloud_pad|>"
        )
        self.im_end_token_id = self.qwen_processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.im_start_token_id = self.qwen_processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

        # FREEZE_QWEN and USE_LORA are mutually exclusive.
        assert not (cfg.FREEZE_QWEN and cfg.USE_LORA), (
            "FREEZE_QWEN and USE_LORA cannot both be enabled."
        )

        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            pointcloud_token_id=pointcloud_token_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.this_device,
            attn_implementation="sdpa",
        )

        if cfg.FREEZE_QWEN:
            for param in self.qwen_model.parameters():
                param.requires_grad = False
        elif cfg.USE_LORA:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                r=cfg.LORA_RANK,
                lora_alpha=cfg.LORA_ALPHA,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "qkv", "visual.blocks.*.attn.proj",
                    "gate_proj", "up_proj", "down_proj", "lm_head",
                ],
                lora_dropout=cfg.LORA_DROPOUT,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.qwen_model = get_peft_model(self.qwen_model, lora_config)

        self.qwen_3d_encoder = Qwen3DEncoder(
            model=self.qwen_model.visual,
            processor=self.qwen_processor,
            voxel_size=0.05,
            feature_dim=self.qwen_model.config.hidden_size,
            min_points_per_voxel=1,
            device=self.this_device,
        )

        # Projection from Qwen hidden dim into the mask decoder's feature dim.
        qwen_hidden_dim = self.qwen_model.config.hidden_size
        decoder_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        self.connector = nn.Sequential(
            nn.Linear(qwen_hidden_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
        )
        self.text_connector = nn.Sequential(
            nn.Linear(qwen_hidden_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
        )

        self.mask_decoder = mask_decoder
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        assert size_divisibility >= 0, size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_feature_levels = 3

        self.decoder_3d = decoder_3d
        self.supervise_sparse = supervise_sparse
        self.eval_sparse = eval_sparse
        self.cfg = cfg

        self.tokenizer = self.qwen_processor.tokenizer

        if self.cfg.LOG_GRADIENTS:
            assert self.cfg.USE_WANDB
            wandb.watch(self, log="all", log_freq=1)

    @classmethod
    def from_config(cls, cfg):
        from qwen3d.modeling.transformer_decoder.simple_mask_decoder import (
            build_transformer_decoder,
        )
        mask_decoder = build_transformer_decoder(
            cfg,
            cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            mask_classification=True,
        )

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        box_weight = cfg.MODEL.MASK_FORMER.BOX_WEIGHT
        giou_weight = cfg.MODEL.MASK_FORMER.GIOU_WEIGHT
        generation_weight = cfg.MODEL.MASK_FORMER.GENERATION_WEIGHT

        matching_class_weight = class_weight if cfg.MATCHING_CLASS_WEIGHT is None else cfg.MATCHING_CLASS_WEIGHT
        matching_dice_weight = dice_weight if cfg.MATCHING_DICE_WEIGHT is None else cfg.MATCHING_DICE_WEIGHT
        matching_mask_weight = mask_weight if cfg.MATCHING_MASK_WEIGHT is None else cfg.MATCHING_MASK_WEIGHT
        matching_box_weight = box_weight if cfg.MATCHING_BOX_WEIGHT is None else cfg.MATCHING_BOX_WEIGHT
        matching_giou_weight = giou_weight if cfg.MATCHING_GIOU_WEIGHT is None else cfg.MATCHING_GIOU_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=matching_class_weight,
            cost_mask=matching_mask_weight,
            cost_dice=matching_dice_weight,
            cost_bbox=matching_box_weight,
            cost_giou=matching_giou_weight,
            cost_mask_det=cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
            cost_class_det=cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
            cost_dice_det=cfg.MODEL.MASK_FORMER.DICE_WEIGHT,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_2D,
            supervise_sparse=cfg.MODEL.SUPERVISE_SPARSE,
            cfg=cfg,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_bbox": box_weight,
            "loss_giou": giou_weight,
            "loss_generation": generation_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        if cfg.USE_BOX_LOSS:
            losses.append("bboxes")

        if cfg.GENERATION:
            losses.append("generation")

        criterion_fn = VideoSetCriterion

        criterion = criterion_fn(
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_2D,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            supervise_sparse=cfg.MODEL.SUPERVISE_SPARSE,
            cfg=cfg,
        )

        return {
            "mask_decoder": mask_decoder,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) if len(cfg.DATASETS.TRAIN) > 0 else MetadataCatalog.get(cfg.DATASETS.TRAIN_2D[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "decoder_3d": cfg.MODEL.DECODER_3D,
            "supervise_sparse": cfg.MODEL.SUPERVISE_SPARSE,
            "eval_sparse": cfg.TEST.EVAL_SPARSE,
            "cfg": cfg,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def load_cached_image_features(self, batched_inputs):
        """Return cached, pre-voxelized image feature cloud if present, else None.

        Caching is done per-image by the dataset mapper; if all images for the
        scene have cached features, they are concatenated and returned.
        """
        if batched_inputs[0].get("qwen_fcs", None) is None:
            return None
        return torch.cat(batched_inputs[0]["qwen_fcs"], dim=0).to(self.device)

    def encode_and_cache_image_features(self, batched_inputs, images, feature_paths):
        """Encode images one-by-one with the Qwen visual encoder, optionally
        writing each per-image feature to disk for future runs.

        Returns (combined_featurecloud, combined_grid_thw).
        """
        all_featureclouds = []
        all_grid_thws = []
        with torch.no_grad():
            for i, (image, save_path) in enumerate(zip(images, feature_paths)):
                single_fc, single_thw = self.qwen_3d_encoder.process_scene(
                    images=[image],
                    text=None,
                    batch_size=1,
                )
                if save_path is not None and batched_inputs[0]["actual_decoder_3d"]:
                    torch.save(
                        {
                            "featurecloud": single_fc.cpu(),
                            "image_index": i,
                            "voxel_size": self.qwen_3d_encoder.voxel_size,
                        },
                        save_path,
                    )
                all_featureclouds.append(single_fc)
                all_grid_thws.append(single_thw)

            combined_featurecloud = torch.cat(all_featureclouds, dim=0).to(self.device)
            combined_grid_thws = torch.cat(all_grid_thws, dim=0).to(self.device)
        return combined_featurecloud, combined_grid_thws

    def encode_scene_3d_cached(self, batched_inputs):
        """Produce the per-point feature cloud for a single scene.

        Falls back to a cached feature cloud if the dataset mapper has populated
        it; otherwise runs the Qwen visual encoder and (optionally) caches the
        per-image outputs to disk.

        Returns:
            featurecloud: (N, D) tensor of per-point Qwen features.
            image_grid_thw: per-image (T, H, W) grid sizes used by Qwen's RoPE
                in the 2D path, or None when features were loaded from cache.
        """
        cached = self.load_cached_image_features(batched_inputs)
        if cached is not None:
            return cached, None

        images = []
        for video in batched_inputs:
            for image in video["images"]:
                images.append(image.to(self.device))

        feature_paths = batched_inputs[0].get("qwen_feature_paths", None)
        if feature_paths is None:
            feature_paths = [None] * len(images)

        return self.encode_and_cache_image_features(batched_inputs, images, feature_paths)
    

    def load_3d_data(self, batched_inputs, images_shape):
        valids = None
        multiview_data = None
        bs, _ = images_shape[:2]
        
        multiview_data = {}
        multiview_data["multi_scale_xyz"] = [
            torch.stack(
                [batched_inputs[i]["multi_scale_xyz"][j] for i in range(bs)], dim=0
            ).to(self.device)
            for j in range(len(batched_inputs[0]["multi_scale_xyz"]))
        ]

        voxel_size = self.cfg.INPUT.VOXEL_SIZE[::-1]
        if self.cfg.INPUT.VOXELIZE:
            multiview_data["multi_scale_p2v"] = multiscsale_voxelize(
                multiview_data["multi_scale_xyz"], voxel_size
            )
        return valids, multiview_data

    def upsample_pred_masks(
        self,
        mask_pred_results,
        batched_inputs,
        multiview_data,
        shape,
        downsample=False,
        interp="bilinear",
    ):
        bs, v, H_padded, W_padded = shape
        assert bs == 1
        if interp == "trilinear":
            target_xyz = batched_inputs["original_xyz"][None].to(self.device)
            if downsample:
                target_xyz = (
                    F.interpolate(
                        target_xyz[0].permute(0, 3, 1, 2),
                        scale_factor=0.5,
                        mode="nearest",
                    )
                    .permute(0, 2, 3, 1)
                    .reshape(
                        bs,
                        v,
                        target_xyz.shape[2] // 2,
                        target_xyz.shape[3] // 2,
                        target_xyz.shape[4],
                    )
                )
            target_p2v = torch.arange(
                target_xyz.flatten(1, 3).shape[1], device=self.device
            )[None]
            source_xyz = multiview_data["multi_scale_xyz"]
            source_p2v = multiview_data["multi_scale_p2v"]

            mask_pred_results = mask_pred_results[:, source_p2v][None]

            mask_pred_results = mask_pred_results.permute(0, 2, 1)
            source_xyz = source_xyz.flatten(0, 2)[None]
            source_p2v = source_p2v[None]
            B, _, Q = mask_pred_results.shape

            mask_pred_results = (
                interpolate_feats_3d(
                    mask_pred_results,
                    source_xyz,
                    source_p2v,
                    target_xyz,
                    target_p2v,
                    shape=[bs, v],
                    num_neighbors=self.cfg.INTERP_NEIGHBORS,
                    voxelize=True,
                )
                .reshape(
                    target_xyz.shape[1], Q, target_xyz.shape[-3], target_xyz.shape[-2]
                )
                .permute(1, 0, 2, 3)
                .to(mask_pred_results.dtype)
            )
        elif interp == "bilinear":
            Q, N, H, W = mask_pred_results.shape
            img_size = (H_padded, W_padded)
            if downsample:
                img_size = (img_size[0] // 2, img_size[1] // 2)

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(img_size[0], img_size[1]),
                mode="bilinear",
                align_corners=False,
            )
        else:
            raise NotImplementedError(
                f"interp must be either trilinear or bilinear, got {interp}"
            )

        return mask_pred_results

    def export_pred_benchmark(self, processed_results, scene_name, dataset_name):
        if "scannet200" in dataset_name:
            learning_map_inv = SCANNET200_LEARNING_MAP_INV
            root_path = "/path/to/language_grounding/benchmark_evaluations/scannet200_80_4"
        else:
            root_path = "/path/to/language_grounding/benchmark_evaluations/scannet_52_10.5"
            learning_map_inv = LEARNING_MAP_INV

        base_path = f"{root_path}/instance_evaluation"
        pred_mask_path = f"{base_path}/pred_mask"

        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        file_name = scene_name
        with open(f"{base_path}/{file_name}.txt", "w") as fout:
            real_id = -1
            pred_classes = (
                processed_results["instances_3d"]["pred_classes"].cpu().numpy()
            )
            scores = processed_results["instances_3d"]["pred_scores"].cpu().numpy()
            pred_masks = processed_results["instances_3d"]["pred_masks"].cpu().numpy()
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                pred_class = learning_map_inv[pred_class]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                np.savetxt(
                    f"{pred_mask_path}/{file_name}_{real_id}.txt", mask, fmt="%d"
                )
                fout.write(
                    f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n"
                )

        # for semantic segmentation
        base_path = f"{root_path}/semantic_evaluation"
        Path(base_path).mkdir(parents=True, exist_ok=True)

        pred_mask_path = f"{base_path}/{scene_name}.txt"

        with open(pred_mask_path, "w") as fout:
            pred_mask = processed_results["semantic_3d"].cpu().numpy()
            pred_mask = np.array([learning_map_inv[x + 1] for x in pred_mask])
            np.savetxt(pred_mask_path, pred_mask, fmt="%d")

        torch.cuda.empty_cache()

    def eval_ghost(
        self,
        mask_cls_results,
        mask_pred_results,
        batched_inputs,
        scannet_gt_target_dicts,
        scannet_p2v,
        num_classes,
        scannet_idxs,
        segments,
        scannet_all_masks_batched=None,
    ):
        processed_results = []

        if (not self.cfg.USE_GT_MASKS or 'ref' not in batched_inputs[0]['dataset_name']) and self.cfg.USE_SEGMENTS:
            mask_pred_results = voxel_map_to_source(
                mask_pred_results.permute(0, 2, 1), segments if not self.cfg.USE_GT_MASKS else scannet_all_masks_batched
            ).permute(0, 2, 1)

        pred_masks = mask_pred_results
        for i, pred_mask in enumerate(pred_masks):
            if self.cfg.USE_GT_MASKS and 'ref' in batched_inputs[i]['dataset_name']:
                max_valid_point = len(scannet_gt_target_dicts[i]['all_scannet_masks'].unique())
                pred_mask = pred_mask[:, :max_valid_point]
            else:
                pred_mask = pred_mask[:, scannet_p2v[i]]

                # remove padding
                max_valid_point = scannet_gt_target_dicts[i]["max_valid_points"]
                pred_mask = pred_mask[:, :max_valid_point]

            if self.cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                if 'ref' in batched_inputs[i]['dataset_name']:
                    processed_3d = {
                        "pred_scores": mask_cls_results[i],
                        "pred_masks": pred_mask,
                        "scannet_idxs": scannet_idxs[i] if len(scannet_idxs) > 0 else None,
                        "reduced_scene_id_to_original_id": scannet_gt_target_dicts[i]["reduced_scene_id_to_original_id"] if self.cfg.USE_GT_MASKS else None,
                    }
                else:
                    processed_3d = self.inference_scannet_ghost(
                        pred_mask, mask_cls_results[i], num_classes=num_classes
                    )

                if "test" not in batched_inputs[i]["dataset_name"]:
                    processed_3d["scannet_gt_masks"] = scannet_gt_target_dicts[i][
                        "masks"
                    ]
                    processed_3d["scannet_gt_classes"] = (
                        scannet_gt_target_dicts[i]["labels"] + 1
                    )
                    processed_3d["max_valid_points"] = max_valid_point
                processed_3d = {"instances_3d": processed_3d}

            if self.cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                semantic_r = retry_if_cuda_oom(self.inference_scannet_ghost_semantic)(
                    mask_cls_results[i], pred_mask
                )
                processed_3d["semantic_3d"] = semantic_r

            if self.cfg.MATTERPORT_ALL_CLASSES_TO_21:
                matterport_all_classes_to_21 = torch.tensor(
                    list(MATTERPORT_ALL_CLASSES_TO_21.values()), device=pred_mask.device
                )
                processed_3d["instances_3d"][
                    "pred_classes"
                ] = matterport_all_classes_to_21[
                    processed_3d["instances_3d"]["pred_classes"] - 1
                ]
                processed_3d["semantic_3d"] = (
                    matterport_all_classes_to_21[processed_3d["semantic_3d"]] - 1
                )

            processed_results.append(processed_3d)

            if self.cfg.EXPORT_BENCHMARK_DATA:
                self.export_pred_benchmark(
                    processed_results[-1],
                    batched_inputs[i]["file_name"].split("/")[-3],
                    dataset_name=batched_inputs[i]["dataset_name"],
                )
                return None

            if self.cfg.VISUALIZE:
                self.visualize_pred_on_scannet(
                    batched_inputs[i],
                    processed_results[i],
                    scannet_gt_target_dicts,
                    index=i,
                    scannet_idxs=scannet_idxs[i] if len(scannet_idxs) > 0 else None,
                )
        return processed_results

    def eval_normal(
        self,
        mask_cls_results,
        mask_pred_results,
        batched_inputs,
        images,
        shape,
        num_classes,
        decoder_3d,
        multiview_data,
        actual_decoder_3d=False,
    ):
        _, v, H_padded, W_padded = shape
        processed_results = []

        for i, (
            mask_cls_result,
            mask_pred_result,
            input_per_image,
            image_size,
        ) in enumerate(
            zip(mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes)
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            shape_ = [1, v, H_padded, W_padded]

            multiview_data_ = None
            if multiview_data is not None:
                multiview_data_ = {}
                multiview_data_["multi_scale_xyz"] = multiview_data["multi_scale_xyz"][
                    -1
                ][i]
                if self.cfg.INPUT.VOXELIZE:
                    multiview_data_["multi_scale_p2v"] = multiview_data[
                        "multi_scale_p2v"
                    ][-1][i]

            if self.eval_sparse:
                valids = input_per_image.get("valids")
                valids = torch.stack(valids).reshape(v, height, width)

            processed_results.append({})

            if self.cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON and decoder_3d:
                semantic_r = self.inference_video_semantic(
                    mask_cls_result,
                    mask_pred_result,
                    image_size,
                    valids if self.eval_sparse else None,
                    batched_inputs[i],
                    multiview_data_,
                    shape_,
                )

                processed_results[-1]["semantic_3d"] = semantic_r
               
            output_img_size = None

            if "coco" in input_per_image["dataset_name"] or "sam" in input_per_image['dataset_name']:
                output_img_size = [
                    input_per_image.get("height"),
                    input_per_image.get("width"),
                ]
                height, width = image_size

            if self.cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                instance_r = self.inference_video(
                    mask_cls_result,
                    mask_pred_result,
                    height,
                    width,
                    valids if self.eval_sparse else None,
                    decoder_3d=decoder_3d,
                    num_classes=num_classes,
                    batched_inputs=batched_inputs[i],
                    multiview_data=multiview_data_,
                    shape=shape_,
                    output_img_size=output_img_size,
                    actual_decoder_3d=actual_decoder_3d,
                )

                if decoder_3d:
                    processed_results[-1]["instances_3d"] = instance_r["3d"]

                if not decoder_3d or not actual_decoder_3d:
                    processed_results[-1]["instances"] = instance_r["2d"]
        return processed_results
    
    def duplicate_ref(
        self, batched_inputs, sr3d_data, valids, segments, multiview_data
    ):
        bs = len(sr3d_data)
        batched_inputs = [copy.copy(batched_inputs[0]) for i in range(bs)]
        for i in range(bs):
            batched_inputs[i]['sr3d_data'] = [sr3d_data[i]]
        if valids is not None:
            valids = valids.repeat(bs, 1, 1, 1)
        if segments is not None:
            segments = segments.repeat(bs, 1, 1)
        multiview_data['multi_scale_xyz'] = [multiview_data['multi_scale_xyz'][i].repeat(bs, 1, 1, 1, 1) for i in range(len(multiview_data['multi_scale_xyz']))]
        multiview_data['multi_scale_p2v'] = [multiview_data['multi_scale_p2v'][i].repeat(bs, 1) for i in range(len(multiview_data['multi_scale_p2v']))]
        return valids, segments, multiview_data, batched_inputs

    def add_boxes_to_targets(self, targets, pcs):
        """
            boxes are in the format [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        for i, target in enumerate(targets):
            assert not self.cfg.USE_GT_MASKS
            masks = target['masks'].to(torch.bool)
            pc = pcs[i, :masks.shape[1]]
            object_bbox = torch.tensor(
                [pc[mask].min(0)[0].tolist() + pc[mask].max(0)[0].tolist() if mask.sum() > 1 else [0.0, 0.0, 0.0, 1e-2, 1e-2, 1e-2] for mask in masks], device=masks.device
            )
            targets[i]['boxes'] = object_bbox
        return targets   
    
    # TODO: Change this so that bs > 1 is actually proper ([b, idx] should be [b, idx] because each batch may have different sizes so the inputs should actually be lists not a single tensor)
    def random_sample_pointcloud(self, pointcloud, feature_cloud, pixel_indices, point2voxel, target_points):
        """Simple random sampling - very fast but less structure-preserving"""
        batch_size, num_points, _ = pointcloud.shape
        device = pointcloud.device
        
        downsampled_pc = []
        downsampled_features = []
        downsampled_p2v = []
        downsampled_pixel_indices = []
        
        target_points = min(target_points, num_points)
        
        for b in range(batch_size):
            # Random sampling of voxels
            idx = torch.randperm(num_points, device=device)[:target_points]
            
            downsampled_pc.append(pointcloud[b, idx])
            downsampled_features.append(feature_cloud[b, idx])
            downsampled_pixel_indices.append(pixel_indices[b][idx])
            
            # Create mapping from old voxel indices to new voxel indices
            old_to_new_voxel = torch.full((num_points,), -1, dtype=torch.long, device=device)
            old_to_new_voxel[idx] = torch.arange(target_points, device=device)
            
            # Remap point2voxel indices
            new_p2v = old_to_new_voxel[point2voxel[b]]
            
            # Find pixels that map to removed voxels (-1 values)
            invalid_mask = new_p2v == -1
            
            if invalid_mask.any():
                # Get the original point cloud coordinates that need reassignment
                # We need to get these from the original combined point cloud
                # Since we don't have direct access to original coordinates here,
                # we'll use a distance-based approach with the voxel centers
                
                invalid_indices = torch.where(invalid_mask)[0]
                
                # For each invalid pixel, find the nearest remaining voxel
                for invalid_idx in invalid_indices:
                    # Get the original voxel index this pixel was mapped to
                    orig_voxel_idx = point2voxel[b, invalid_idx]
                    orig_voxel_center = pointcloud[b, orig_voxel_idx]  # Original voxel center
                    
                    # Find distances to all remaining voxel centers
                    remaining_voxels = pointcloud[b, idx]  # Downsampled voxel centers
                    distances = torch.norm(remaining_voxels - orig_voxel_center, dim=1)
                    
                    # Assign to nearest remaining voxel
                    nearest_voxel_idx = torch.argmin(distances)
                    new_p2v[invalid_idx] = nearest_voxel_idx
            
            downsampled_p2v.append(new_p2v)
        
        return (
            torch.stack(downsampled_pc), 
            torch.stack(downsampled_features),
            torch.stack(downsampled_p2v),
            torch.stack(downsampled_pixel_indices)
        )

    def upsample_point_features(self, source_pc, source_features, target_pc, k_neighbors=3):
        """
        Upsample point cloud features from a sparse point cloud to a denser target point cloud.
        
        Args:
            source_pc: Source sparse point cloud [B, N_source, 3]
            source_features: Source point features [B, N_source, C]
            target_pc: Target dense point cloud [B, N_target, 3]
            k_neighbors: Number of nearest neighbors for interpolation
            
        Returns:
            Upsampled features [B, N_target, C]
        """
        batch_size = source_pc.shape[0]
        
        source_batch_offset =  (
            (torch.arange(batch_size, dtype=torch.int32, device=source_pc.device) + 1) * source_pc.shape[1]
        )
        target_batch_offset = (
            (torch.arange(batch_size, dtype=torch.int32, device=target_pc.device) + 1) * target_pc.shape[1]
        )
        import libs.pointops2.functions.pointops as pointops
        upsampled = pointops.interpolation(
            source_pc.flatten(0, 1),
            target_pc.flatten(0, 1),
            source_features.flatten(0, 1),
            source_batch_offset,
            target_batch_offset,
            k=k_neighbors,
        )
        upsampled = rearrange(upsampled, '(b n) c -> b c n', b=batch_size)
        return upsampled


    def get_target_sentence_spans(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Returns (starts, ends) where each is [B], giving the start (inclusive) and end (exclusive)
        indices of the target_sentence tokens for each batch item.
        Span is defined as tokens between the last <|vision_end|> and the next <|im_end|>.
        """
        im_end_id = self.im_end_token_id  # <|im_end|>
        vision_end_id = self.qwen_processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        B, _ = input_ids.shape
        starts = torch.full((B,), -1, dtype=torch.long, device=input_ids.device)
        ends = torch.full((B,), -1, dtype=torch.long, device=input_ids.device)

        for b in range(B):
            seq = input_ids[b]
            vis_ends = (seq == vision_end_id).nonzero(as_tuple=False).flatten()
            if vis_ends.numel() == 0:
                continue
            last_vis_end = vis_ends[-1].item()

            im_ends = (seq == im_end_id).nonzero(as_tuple=False).flatten()
            im_ends_after = im_ends[im_ends > last_vis_end]
            if im_ends_after.numel() == 0:
                continue

            starts[b] = last_vis_end + 1
            ends[b] = im_ends_after[0]

        return starts, ends

    def forward(self, batched_inputs):
        """Run one forward pass on a single-scene batch.

        Args:
            batched_inputs (list[dict]): List of length ``bs == 1`` produced by
                the dataset mapper. The relevant keys consumed below are:

                - ``decoder_3d`` (bool): batch is routed through the 3D path.
                - ``actual_decoder_3d`` (bool): scene is genuinely 3D (vs.
                  pseudo-3D augmentation of a 2D sample).
                - ``text_only`` (bool, optional): text-only QA batch with no
                  images / point cloud.
                - ``images`` (list[Tensor[3, H, W]]): per-frame images.
                - ``new_h`` / ``new_w`` (int): feature-map H/W produced by the
                  Qwen visual encoder for each image.
                - ``multi_scale_xyz`` (list[Tensor]): per-scale per-pixel xyz
                  coordinates; ``multi_scale_xyz[1]`` matches the encoder's
                  feature-map resolution.
                - ``scannet_coords`` / ``scannet_color`` / ``scannet_labels`` /
                  ``scannet_segments``: full-scene point cloud (used by the
                  ghost-points evaluation path).
                - ``sr3d_data`` (list[dict]): grounding annotation (referring
                  expression, target id, anchor ids, positive map, etc.).
                - ``dataset_name`` / ``num_classes`` / ``all_classes``: dataset
                  metadata used to build per-batch class prompts.

        Returns:
            During training: a dict of scalar losses.
            During eval: ``processed_results`` (list[dict]), and if a
            text-generation dataset is active, ``(generation_language,
            processed_results)``.
        """
        assert torch.get_autocast_dtype(device_type="cuda") == torch.bfloat16, (
            "We want to train this model in bfloat16"
        )

        # NOTE: bs == 1 is enforced below; the sum / >0 idiom is kept so this
        # also works correctly should bs > 1 ever be enabled.
        decoder_3d = (
            sum(b["decoder_3d"] for b in batched_inputs) > 0
        )
        actual_decoder_3d = batched_inputs[0]["actual_decoder_3d"]
        if self.training and self.cfg.PSEUDO_2D_AUG and decoder_3d and not actual_decoder_3d:
            if random.random() > 0.5:
                decoder_3d = False

        bs = len(batched_inputs)
        assert bs == 1, "Currently only batch size of 1 is supported for qwen3D design"

        if not batched_inputs[0].get("text_only", False):
            images = []
            for video in batched_inputs:
                for image in video["images"]:
                    images.append(image.to(self.device))
            v = len(batched_inputs[0]["images"])

            featurecloud, image_grid_thw = self.encode_scene_3d_cached(batched_inputs)

            # All inputs in a batch share the same feature-map (new_h, new_w).
            hardcoded_feature_H = batched_inputs[0]["new_h"]
            hardcoded_feature_W = batched_inputs[0]["new_w"]

            # NOTE: images are still normalized for the dataset mapper / viz
            # tooling, even though they are not fed into a separate visual
            # backbone (Qwen's encoder consumes raw images).
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            H_padded, W_padded = images.tensor.shape[-2:]
            valids, segments = None, None

            if decoder_3d:
                valids, multiview_data = self.load_3d_data(
                    batched_inputs,
                    images_shape=[bs, v, H_padded, W_padded],
                )

                pointcloud = multiview_data["multi_scale_xyz"][1].reshape(-1, 3).to(self.device)
                point2voxel = multiview_data["multi_scale_p2v"][1].squeeze(0).to(self.device)
                assert featurecloud.shape[0] == pointcloud.shape[0] == point2voxel.shape[0], (
                    f"featurecloud.shape[0]={featurecloud.shape[0]}, "
                    f"pointcloud.shape[0]={pointcloud.shape[0]}, "
                    f"point2voxel.shape[0]={point2voxel.shape[0]}"
                )

                featurecloud = scatter_mean(featurecloud, point2voxel, dim=0).unsqueeze(0)
                pointcloud = scatter_mean(pointcloud, point2voxel, dim=0).unsqueeze(0)
                point2voxel = point2voxel.unsqueeze(0)

                pixel_indices = sample_2D_indices(
                    hardcoded_feature_H,
                    hardcoded_feature_W,
                    [multiview_data["multi_scale_p2v"][i * 2 + 1][0] for i in range(bs)],
                    [len(batched_inputs[i]["images"]) for i in range(bs)],
                )
            else:
                featurecloud = featurecloud.unsqueeze(0)


            (
                scannet_pc,
                scannet_gt_target_dicts,
                scannet_p2v,
                scannet_segments_batched,
                scannet_all_masks_batched,
            ) = (None, None, None, None, None)

            if self.cfg.USE_GHOST_POINTS and decoder_3d:
                full_scene_dataset = batched_inputs[0]["full_scene_dataset"]
                (
                    scannet_pc,
                    scannet_p2v,
                    scannet_gt_target_dicts,
                    scannet_idxs,
                    scannet_segments_batched,
                    scannet_all_masks_batched,
                ) = load_scannet_data(
                    self.cfg,
                    batched_inputs,
                    multiview_data,
                    do_knn=self.training or not full_scene_dataset or self.cfg.FORCE_SUBSAMPLE,
                    images=images,
                    shape=[bs, v, H_padded, W_padded],
                    device=self.device,
                    is_training=self.training,
                    tokenizer=self.tokenizer,
                    our_upsampled_pc = pointcloud,
                    pixel_mean=self.pixel_mean,
                    pixel_std=self.pixel_std,
                )

            if self.cfg.USE_GHOST_POINTS and decoder_3d:
                targets = scannet_gt_target_dicts
            else:
                targets = prepare_targets(
                    self.cfg,
                    batched_inputs,
                    images,
                    valids,
                    dataset_names=[batched_input["dataset_name"] for batched_input in batched_inputs],
                    is_training=self.training,
                    tokenizer=self.tokenizer,
                    device=self.device,
                )

            captions = None
            if self.cfg.MODEL.OPEN_VOCAB:
                captions = [targets[i]["text_caption"] for i in range(len(targets))]
                if "num_classes" in batched_inputs[0].keys():
                    num_classes = (
                        max(b["num_classes"] for b in batched_inputs) + 1
                    )

            assert len(captions) == 1, "len(captions) should be 1 for now"
            target_sentence = (
                captions
                if captions is not None
                else [targets[i]["text_caption"] for i in range(len(targets))]
            )
            # Empty captions break the chat template; substitute a single space.
            target_sentence = [s if s != "" else " " for s in target_sentence]

            if self.training and decoder_3d:
                (
                    pointcloud,
                    featurecloud,
                    point2voxel,
                    pixel_indices,
                ) = self.random_sample_pointcloud(
                    pointcloud, featurecloud, pixel_indices, point2voxel,
                    target_points=4096,
                )
            if self.training and self.cfg.GENERATION:
                if actual_decoder_3d:
                    answers = [
                        random.choice(sample["sr3d_data"][0]["answers"])
                        if "sr3d_data" in sample and "answers" in sample["sr3d_data"][0]
                        else ""
                        for sample in batched_inputs
                    ]
                else:
                    answers = [
                        targets[i].get("answer", "") for i in range(len(targets))
                    ]
            else:
                answers = ["" for _ in range(bs)]
        else:
            target_sentence = [b["text_caption"] for b in batched_inputs]
            answers = [b["answer"] for b in batched_inputs]

        # Build the chat-formatted prompt for each example. The template differs
        # by task: text-only / 3D scene / 2D image; train vs. generation.
        text_input = []
        for i in range(bs):
            if batched_inputs[0].get("text_only", False):
                template = TEXT_INPUT_TRAIN if self.training else TEXT_INPUT_GEN
            elif decoder_3d:
                template = TEXT_INPUT_TRAIN_3D if self.training else TEXT_INPUT_GEN_3D
            else:
                template = TEXT_INPUT_TRAIN_2D if self.training else TEXT_INPUT_GEN_2D

            if self.training:
                text_input.append(
                    template.format(target_sentence=target_sentence[i], answer=answers[i])
                )
            else:
                text_input.append(template.format(target_sentence=target_sentence[i]))

        inputs = self.qwen_processor.tokenizer(text_input, return_tensors="pt").to(
            self.qwen_model.device
        )

        if not batched_inputs[0].get("text_only", False):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # LoRA wraps the base model an extra level deep.
            if self.cfg.USE_LORA:
                embed_tokens = self.qwen_model.base_model.model.model.embed_tokens
            else:
                embed_tokens = self.qwen_model.model.embed_tokens
            inputs_embeds = embed_tokens(input_ids)

            # Locate the placeholder token in the prompt that we will replace
            # with the per-point (3D) or per-patch (2D) Qwen features.
            placeholder_token_id = (
                self.qwen_model.config.pointcloud_token_id
                if decoder_3d
                else self.qwen_model.config.image_token_id
            )
            pc_mask = input_ids == placeholder_token_id
            assert pc_mask.sum() == 1, (
                "There should be exactly one placeholder (point cloud / image) "
                "token per input for now"
            )
            pc_pos = pc_mask.int().argmax(dim=1)  # [B]

            # Splice the visual / point features into the token sequence.
            new_attention_masks = []
            new_inputs_embeds = []
            new_input_ids = []
            for b in range(bs):
                idx = pc_pos[b].item()
                n_pts = featurecloud[b].shape[0]

                new_attention_masks.append(
                    torch.cat(
                        [
                            attention_mask[b, :idx],
                            attention_mask.new_ones(n_pts),
                            attention_mask[b, idx + 1 :],
                        ],
                        dim=0,
                    )
                )
                new_inputs_embeds.append(
                    torch.cat(
                        [inputs_embeds[b, :idx], featurecloud[b], inputs_embeds[b, idx + 1 :]],
                        dim=0,
                    )
                )
                new_input_ids.append(
                    torch.cat(
                        [
                            input_ids[b, :idx],
                            input_ids.new_full((n_pts,), placeholder_token_id),
                            input_ids[b, idx + 1 :],
                        ],
                        dim=0,
                    )
                )
            inputs["input_ids"] = torch.stack(new_input_ids, dim=0)
            inputs["padding_mask"] = torch.stack(new_attention_masks, dim=0)
            inputs["inputs_embeds"] = torch.stack(new_inputs_embeds, dim=0)

            if decoder_3d:
                inputs["rope_type"] = self.cfg.ROPE_TYPE
                inputs["pointcloud_pixel_pos"] = pixel_indices
                del pixel_indices
                inputs["pointcloud_xyz"] = pointcloud
            else:
                assert image_grid_thw is not None, (
                    "image_grid_thw should not be None for the 2D decoder path"
                )
                inputs["rope_type"] = "default"
                inputs["image_grid_thw"] = image_grid_thw.view(bs * v, 3)
        else:
            inputs["padding_mask"] = torch.ones(
                1,
                inputs["input_ids"].size(1),
                dtype=torch.bool,
                device=inputs["input_ids"].device,
            )

        inputs["use_causal_mask"] = self.cfg.CAUSAL_MASK

        # Locate the answer span in the prompt: answer_start is the first
        # answer token (inclusive), answer_end is the last (inclusive).
        # The "+3" skips the assistant role / start-of-turn tokens emitted by
        # the chat template after the final <|im_start|>.
        answer_start = (
            inputs["input_ids"].size(1)
            - 1
            - (inputs["input_ids"].flip(1) == self.im_start_token_id).float().argmax(1)
        ) + 3
        inputs["answer_start"] = answer_start

        generation = self.cfg.GENERATION and (
            batched_inputs[0].get("do_generate", False)
            or batched_inputs[0].get("generate_only", False)
        )
        if batched_inputs[0].get("generate_only", False) or batched_inputs[0].get(
            "do_generate", False
        ):
            assert self.cfg.GENERATION, (
                "generate_only/do_generate is True but cfg.GENERATION is False"
            )

        if self.training or not generation:
            inputs["output_hidden_states"] = True
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100, dtype=torch.long)
            answer_end = (
                inputs["input_ids"].size(1)
                - 1
                - (inputs["input_ids"].flip(1) == self.im_end_token_id).float().argmax(1)
            ) + 1

            B, L = inputs["input_ids"].shape
            positions = torch.arange(L, device=inputs["input_ids"].device).unsqueeze(0)
            mask = (positions >= answer_start[:, None]) & (positions <= answer_end[:, None])
            inputs["labels"] = torch.where(mask, inputs["input_ids"], inputs["labels"])

            if not self.cfg.CAUSAL_MASK:
                # Build a per-example 4D attention mask: bidirectional on the
                # prompt (so points can attend to each other) and causal over
                # the answer span.
                full_padding_mask = build_full_mask(
                    inputs["padding_mask"], inputs["padding_mask"].device
                )
                causal_mask = torch.tril(
                    torch.ones(
                        (L, L),
                        dtype=torch.bool,
                        device=inputs["attention_mask"].device,
                    )
                )
                for b in range(bs):
                    full_padding_mask[b, 0, : answer_start[b], answer_start[b] :] = False
                    full_padding_mask[b, 0, answer_start[b] :, answer_start[b] :] = (
                        causal_mask[answer_start[b] :, answer_start[b] :]
                    )
                inputs["attention_mask"] = torch.where(
                    full_padding_mask,
                    0.0,
                    torch.finfo(self.qwen_model.dtype).min,
                )
            else:
                inputs["attention_mask"] = inputs["padding_mask"]

            inputs["generate_mode"] = generation
            text_head_outputs = self.qwen_model(**inputs)
            text_output_features = text_head_outputs.hidden_states[-1]
            text_loss = text_head_outputs.loss
        else:
            inputs["use_cache"] = True
            if not batched_inputs[0].get("text_only", False):
                inputs["attention_mask"] = inputs["padding_mask"]

            gen_kwargs = dict(self.cfg.GEN_KWARGS)
            text_generate = self.qwen_model.generate(
                **inputs,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **gen_kwargs,
            )
            text_output_ids = text_generate.sequences
            answer_ids = [text_output_ids[b, answer_start:] for b in range(bs)]
            text_decoded = self.qwen_processor.tokenizer.batch_decode(
                answer_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            text_output_features = text_generate.hidden_states[0][-1]
            if not isinstance(text_decoded, list):
                text_decoded = [text_decoded]

        # Generation-only datasets (text benchmarks) short-circuit before the
        # mask decoder.
        if batched_inputs[0].get("generate_only", False):
            if self.training:
                losses = {}
                if text_loss is not None and not torch.isnan(text_loss).any():
                    losses["loss_generation"] = text_loss

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        losses.pop(k)

                if (
                    self.cfg.MULTI_TASK_TRAINING
                    and not self.cfg.IID_MULTITASK_TRAINING
                    and not (self.cfg.GRAD_ACCUMULATION_STEPS > 1)
                ):
                    if actual_decoder_3d:
                        losses["loss_3d"] = sum(losses.values())
                    else:
                        losses["loss_2d"] = sum(losses.values())
                return losses
            return text_decoded, None

        # Extract point / text features from the LLM's last hidden state, then
        # project them into the mask decoder's feature dimension.
        contextualized_point_features = []
        qwen_text_features = []
        target_starts, target_ends = self.get_target_sentence_spans(inputs["input_ids"])
        for b in range(bs):
            start = pc_pos[b].item()
            contextualized_point_features.append(
                text_output_features[b][start : start + featurecloud[b].shape[0]]
            )
            qwen_text_features.append(text_output_features[b][target_starts[b] : target_ends[b]])
        contextualized_point_features = torch.stack(contextualized_point_features, dim=0)
        qwen_text_features = torch.stack(qwen_text_features, dim=0)

        del featurecloud
        contextualized_point_features = self.connector(contextualized_point_features)
        qwen_text_features = self.text_connector(qwen_text_features)

        # Build the mask decoder's input feature tensor. For the 3D path with
        # ghost points we upsample LLM features onto the full scene point
        # cloud; for the 2D path we reshape to a per-view feature map.
        scannet_pc_ = None
        scannet_p2v_ = None
        if self.cfg.USE_GHOST_POINTS and decoder_3d:
            scannet_pc_ = scatter_mean(scannet_pc, scannet_p2v, dim=1)
            scannet_p2v_ = (
                torch.arange(scannet_pc.shape[1], device=scannet_pc.device)
                .unsqueeze(0)
                .repeat(scannet_pc.shape[0], 1)
            )
            qwen_mask_features = self.upsample_point_features(
                source_pc=pointcloud,
                source_features=contextualized_point_features,
                target_pc=scannet_pc_,
                k_neighbors=8,
            )[..., None]
        else:
            c = contextualized_point_features.shape[-1]
            qwen_mask_features = contextualized_point_features.permute(0, 2, 1).reshape(
                bs * v, c, hardcoded_feature_H, hardcoded_feature_W
            )

        outputs = self.mask_decoder(
            qwen_mask_features,
            shape=[bs, v],
            mask_features_xyz=scannet_pc_ if self.cfg.USE_GHOST_POINTS else None,
            segments=scannet_segments_batched if self.cfg.USE_GHOST_POINTS else segments,
            decoder_3d=decoder_3d,
            actual_decoder_3d=actual_decoder_3d,
            scannet_all_masks_batched=scannet_all_masks_batched,
            max_valid_points=(
                [targets[i]["max_valid_points"] for i in range(len(targets))]
                if self.cfg.USE_GHOST_POINTS and decoder_3d
                else None
            ),
            qwen_3d_text_features=qwen_text_features,
        )

        check_for_nans_in_dict(outputs)
        if outputs is None:
            return None

        if self.training:
            losses = self.criterion(
                outputs,
                targets,
                decoder_3d=decoder_3d,
                actual_decoder_3d=actual_decoder_3d,
            )
            if text_loss is not None and not torch.isnan(text_loss).any():
                losses["loss_generation"] = text_loss

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)

            if (
                self.cfg.MULTI_TASK_TRAINING
                and not self.cfg.IID_MULTITASK_TRAINING
                and not (self.cfg.GRAD_ACCUMULATION_STEPS > 1)
            ):
                # Tag the loss so 2D and 3D batches can be balanced downstream.
                tag = "loss_3d" if actual_decoder_3d else "loss_2d"
                losses[tag] = sum(losses.values())

            return losses

        # ---- Eval path ----
        generation_language = None
        dataset_name = batched_inputs[0]["dataset_name"]
        if "scanqa" in dataset_name or "sqa3d" in dataset_name or "bench" in dataset_name:
            generation_language = text_decoded

        if "sr3d_data" in batched_inputs[0]:
            num_classes = max(
                len(b["sr3d_data"][0]["anchor_ids"]) + 1 for b in batched_inputs
            )
        elif "refcoco" in dataset_name:
            num_classes = 2
        else:
            num_classes = max(b["num_classes"] for b in batched_inputs)

        if self.cfg.MODEL.OPEN_VOCAB:
            outputs["pred_logits"] = outputs["pred_logits"].sigmoid()
            is_grounding = "sr3d_data" in batched_inputs[0] or "refcoco" in dataset_name
            if is_grounding:
                outputs["pred_logits"] = torch.cat(
                    [
                        convert_grounding_to_od_logits_ref(
                            logits=outputs["pred_logits"][i][None],
                            num_class=num_classes + 1,
                            positive_maps=targets[i]["positive_map"],
                            reduce="mean",
                        )
                        for i in range(bs)
                    ]
                )
            else:
                outputs["pred_logits"] = torch.cat(
                    [
                        convert_grounding_to_od_logits(
                            logits=outputs["pred_logits"][i][None],
                            num_class=num_classes + 1,
                            positive_map_od=targets[i]["positive_map_od"],
                            reduce="mean",
                        )
                        for i in range(bs)
                    ]
                )

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        del outputs

        if self.cfg.USE_GHOST_POINTS and actual_decoder_3d:
            processed_results = self.eval_ghost(
                mask_cls_results,
                mask_pred_results,
                batched_inputs,
                scannet_gt_target_dicts,
                scannet_p2v,
                num_classes,
                scannet_idxs,
                scannet_segments_batched,
                scannet_all_masks_batched,
            )
        else:
            processed_results = self.eval_normal(
                mask_cls_results,
                mask_pred_results,
                batched_inputs,
                images,
                [bs, v, H_padded, W_padded],
                num_classes,
                decoder_3d,
                multiview_data if decoder_3d else None,
                actual_decoder_3d=actual_decoder_3d,
            )

        if generation_language is not None:
            return generation_language, processed_results
        return processed_results

    def visualize_pred_on_ours(
        self,
        index,
        images,
        shape,
        input_per_image,
        processed_results,
        targets,
        valids,
        fps_xyz=None,
    ):
        bs, v, H_padded, W_padded = shape
        our_pc = input_per_image["original_xyz"]
        if self.cfg.HIGH_RES_INPUT:
            our_pc = F.interpolate(
                our_pc.float().permute(0, 3, 1, 2), scale_factor=0.5, mode="nearest"
            ).permute(0, 2, 3, 1)
            our_pc = our_pc.cpu().numpy()

            if valids is not None:
                valids = (
                    F.interpolate(
                        valids.float().permute(0, 1, 2).unsqueeze(0),
                        scale_factor=0.5,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .bool()
                )

        if valids is not None:
            our_pc = our_pc[valids]

        else:
            our_pc = our_pc.reshape(-1, 3)

        vis_images = images.tensor * self.pixel_std + self.pixel_mean
        vis_images = vis_images.view(bs, v, 3, H_padded, W_padded)[index]
        if self.cfg.HIGH_RES_INPUT:
            vis_images = F.interpolate(vis_images, scale_factor=0.5, mode="bilinear")

        if valids is not None:
            color = vis_images.permute(0, 2, 3, 1)[valids].cpu().numpy() / 255.0
        else:
            color = vis_images.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy() / 255.0
        color = np.clip(color, 0, 1)

        scene_name = input_per_image["file_name"].split("/")[-3]

        pred_scores = processed_results["instances_3d"]["pred_scores"]
        pred_masks = processed_results["instances_3d"]["pred_masks"]
        pred_labels = processed_results["instances_3d"]["pred_classes"]

        sort_idx = torch.argsort(pred_scores)
        pred_masks = pred_masks.permute(1, 0)[sort_idx].cpu().numpy()
        pred_labels = pred_labels[sort_idx].cpu().numpy()

        # select confident predictions
        pred_scores = pred_scores[sort_idx].cpu().numpy()

        conf = pred_scores > 0.0
        pred_masks = pred_masks[conf]
        pred_labels = pred_labels[conf]
        if fps_xyz is not None:
            fps_xyz = fps_xyz[0][sort_idx][conf].cpu().numpy()

        gt_masks = targets[index]["masks"]
        if self.cfg.HIGH_RES_INPUT:
            gt_masks = (
                F.interpolate(gt_masks.float(), scale_factor=0.5, mode="nearest") > 0.5
            )
        else:
            gt_masks = gt_masks.reshape(-1, H_padded // 4, W_padded // 4)
            gt_masks = F.interpolate(
                gt_masks.float().unsqueeze(0), scale_factor=4.0, mode="nearest"
            )[0].bool()

        if valids is not None:
            gt_masks = gt_masks[:, valids].cpu().numpy()
        else:
            gt_masks = gt_masks.flatten(1)
            gt_masks = gt_masks.cpu().numpy()
        gt_labels = targets[index]["labels"].cpu().numpy()

        valids = np.zeros_like(our_pc[:, 0]).astype(bool)

        # valid_idx = np.random.choice(
        #     np.arange(valids.shape[0]), 200000)
        # valids[valid_idx] = True
        valids[:] = True

        dataset_name = input_per_image["dataset_name"]
        vis_utils.plot_3d_offline(
            our_pc.numpy(),
            color,
            masks=pred_masks,
            valids=valids,
            labels=pred_labels,
            gt_masks=gt_masks,
            gt_labels=gt_labels,
            scene_name=scene_name,
            data_dir="vis_sam_3d",
            mask_classes=None,
            dataset_name=dataset_name,
            fps_xyz=fps_xyz,
        )



    def visualize_pred_on_scannet(
        self, input_per_image, processed_result,
        gt_targets, index, scannet_idxs=None,
        fps_xyz=None
    ):
        pc = input_per_image['scannet_coords'].cpu().numpy()
        if scannet_idxs is not None:
            pc = pc[scannet_idxs.cpu().numpy()]

        color = (input_per_image['scannet_color'] / 255.0).cpu().numpy()
        color = np.clip(color, 0, 1)
        if scannet_idxs is not None:
            color = color[scannet_idxs.cpu().numpy()]

        scene_name = input_per_image['file_name'].split('/')[-3]
        pred_scores = processed_result["instances_3d"]['pred_scores']
        pred_masks = processed_result["instances_3d"]['pred_masks']
        pred_labels = processed_result["instances_3d"]['pred_classes']

        # sort by scores in ascending order
        sort_idx = torch.argsort(pred_scores)
        pred_masks = pred_masks.permute(1, 0)[sort_idx].cpu().numpy()
        pred_labels = pred_labels[sort_idx].cpu().numpy()

        # threshold by scores > 0.5
        pred_scores = pred_scores[sort_idx].cpu().numpy()
        conf = pred_scores > 0.05
        pred_masks = pred_masks[conf]
        pred_labels = pred_labels[conf]

        # whiteboard = pred_labels == 44
        # pred_masks = pred_masks[whiteboard]
        # pred_labels = pred_labels[whiteboard]

        if fps_xyz is not None:
            fps_xyz = fps_xyz[0][sort_idx][conf].cpu().numpy()

        gt_masks = gt_targets[index]['masks'].cpu().numpy()
        if "max_valid_points" in gt_targets[index]:
            max_valid_point = gt_targets[index]["max_valid_points"]
            gt_masks = gt_masks[:, :max_valid_point]

        gt_labels = gt_targets[index]['labels'].cpu().numpy()

        valids = np.ones_like(pc[:, 0]).astype(bool)

        dataset_name = input_per_image['dataset_name']

        vis_utils.plot_3d_offline(
            pc, color, masks=pred_masks, valids=valids,
            labels=pred_labels,
            gt_masks=gt_masks, gt_labels=gt_labels, scene_name=scene_name,
            data_dir=self.cfg.VISUALIZE_LOG_DIR,
            mask_classes=self.cfg.SKIP_CLASSES, dataset_name=dataset_name,
            fps_xyz=fps_xyz
        )


    def prepare_2d(
        self,
        pred_masks,
        img_size,
        labels_per_image,
        scores_per_image,
        batched_inputs=None,
        multiview_data=None,
        shape=None,
        decoder_3d=False,
        output_img_size=None,
    ):
        pred_masks = self.upsample_pred_masks(
            pred_masks,
            batched_inputs,
            multiview_data,
            shape,
            downsample=False,
            interp="trilinear" if decoder_3d else "bilinear",
        )

        context_img_id = pred_masks.shape[1] // 2
        pred_masks = pred_masks[:, context_img_id]

        pred_masks = pred_masks[:, : img_size[0], : img_size[1]]

        if output_img_size is not None:
            pred_masks = F.interpolate(
                pred_masks[None], size=output_img_size, mode="bilinear"
            )[0]

        masks = pred_masks > 0.0
        image_size = masks.shape[-2:]

        mask_scores_per_image = (
            pred_masks.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)

        result_2d = Instances(image_size)
        result_2d.pred_masks = masks
        result_2d.pred_boxes = Boxes(torch.zeros(masks.size(0), 4))
        mask_scores_per_image = (
            pred_masks.sigmoid().flatten(1) * result_2d.pred_masks.flatten(1)
        ).sum(1) / (result_2d.pred_masks.flatten(1).sum(1) + 1e-6)
        result_2d.scores = scores_per_image * mask_scores_per_image
        result_2d.pred_classes = labels_per_image
        return result_2d
    
    def prepare_3d(
        self,
        pred_masks,
        output_height,
        output_width,
        labels_per_image,
        scores_per_image,
        valids=None,
        batched_inputs=None,
        multiview_data=None,
        shape=None,
    ):
        pred_masks = self.upsample_pred_masks(
            pred_masks,
            batched_inputs,
            multiview_data,
            shape,
            downsample=self.cfg.HIGH_RES_INPUT,
            interp="trilinear",
        )

        if valids is not None:
            # downsample valids
            if self.size_divisibility > 1:
                h, w = output_height, output_width
                pad_h = int(
                    np.ceil(h / self.size_divisibility) * self.size_divisibility - h
                )
                pad_w = int(
                    np.ceil(w / self.size_divisibility) * self.size_divisibility - w
                )
                valids = F.pad(valids, (0, pad_w, 0, pad_h), mode="constant", value=0)
            H, W = pred_masks.shape[-2:]
            valids = (
                F.interpolate(valids.float().unsqueeze(0), size=(H, W), mode="nearest")
                .squeeze(0)
                .bool()
            )

        if valids is not None:
            pred_masks = pred_masks[:, valids]

        masks = pred_masks > 0.0
        mask_scores_per_image = (
            pred_masks.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)

        # add +1 to labels as mask3d evals from 1-18
        result_3d = {
            "pred_classes": labels_per_image + 1,
            "pred_masks": masks.flatten(1).permute(1, 0),
            "pred_scores": scores_per_image * mask_scores_per_image,
        }
        return result_3d

    def inference_video(
        self,
        pred_cls,
        pred_masks,
        output_height,
        output_width,
        valids=None,
        decoder_3d=False,
        num_classes=None,
        batched_inputs=None,
        multiview_data=None,
        shape=None,
        output_img_size=None,
        actual_decoder_3d=False,
    ):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        test_topk_per_image = pred_masks.shape[0] 
        
        if self.cfg.MODEL.OPEN_VOCAB and not self.cfg.NON_PARAM_SOFTMAX:
            scores = pred_cls[:, :-1]
        else:
            if self.cfg.NON_PARAM_SOFTMAX:
                scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            elif self.cfg.OPEN_VOCAB_SIGMOID:
                scores = pred_cls[..., :-1].sigmoid()

        skip_classes = self.cfg.SKIP_CLASSES if decoder_3d else self.cfg.SKIP_CLASSES_2D

        if skip_classes is not None:
            skip_classes = torch.tensor(skip_classes, device=self.device) - 1

            # +1 for background class
            keep_class_mask = torch.ones(num_classes, device=self.device)
            keep_class_mask[skip_classes] = 0
            scores = scores[:, keep_class_mask.bool()]
            num_classes -= len(skip_classes)

        
        num_queries = (
            self.num_queries * 2
            if self.cfg.SEPERATE_2D_3D_QUERIES
            else self.num_queries
        )

        labels = (
            torch.arange(num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            test_topk_per_image, sorted=False
        )
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        pred_masks = pred_masks[topk_indices]

        results = {}
        if not decoder_3d or not actual_decoder_3d:
            results["2d"] = self.prepare_2d(
                pred_masks,
                (output_height, output_width),
                labels_per_image,
                scores_per_image,
                batched_inputs,
                multiview_data,
                shape,
                decoder_3d,
                output_img_size,
            )

        if decoder_3d:
            results["3d"] = self.prepare_3d(
                pred_masks,
                output_height,
                output_width,
                labels_per_image,
                scores_per_image,
                valids,
                batched_inputs,
                multiview_data,
                shape,
            )
        return results

    def inference_video_semantic(
        self,
        mask_cls,
        mask_pred,
        image_size=None,
        valids=None,
        batched_inputs=None,
        multiview_data=None,
        shape=None,
    ):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        mask_pred = self.upsample_pred_masks(
            mask_pred,
            batched_inputs,
            multiview_data,
            shape,
            downsample=self.cfg.HIGH_RES_INPUT,
            interp="trilinear",
        )

        if self.cfg.MODEL.OPEN_VOCAB and not self.cfg.NON_PARAM_SOFTMAX:
            mask_cls = mask_cls[..., :-1]
        else:
            if self.cfg.NON_PARAM_SOFTMAX:
                mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            elif self.cfg.OPEN_VOCAB_SIGMOID:
                mask_cls = mask_cls[..., :-1].sigmoid()

        mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qvhw->cvhw", mask_cls, mask_pred).max(0)[1]
        if valids is not None:
            if self.size_divisibility > 1:
                h, w = image_size[0], image_size[1]
                pad_h = int(
                    np.ceil(h / self.size_divisibility) * self.size_divisibility - h
                )
                pad_w = int(
                    np.ceil(w / self.size_divisibility) * self.size_divisibility - w
                )
                valids = F.pad(valids, (0, pad_w, 0, pad_h), mode="constant", value=0)
            H, W = mask_pred.shape[-2:]
            valids = (
                F.interpolate(valids.float().unsqueeze(0), size=(H, W), mode="nearest")
                .squeeze(0)
                .bool()
            )
            semseg = semseg[valids]
        return semseg.reshape(-1)

    def inference_scannet_ghost(self, pred_masks, pred_cls, num_classes):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        test_topk_per_image = pred_masks.shape[
            0
        ]  # 100 #self.cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        
        if self.cfg.MODEL.OPEN_VOCAB:
            scores = pred_cls[:, :-1]
        else:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]

        # num_classes = self.sem_seg_head.num_classes
        if num_classes == 20 and self.cfg.SKIP_CLASSES is not None:
            # because we skip floor and wall for evaluation

            # -1 to 0 index it
            skip_classes = torch.tensor(self.cfg.SKIP_CLASSES, device=self.device) - 1

            # +1 for background class
            keep_class_mask = torch.ones(num_classes, device=self.device)
            keep_class_mask[skip_classes] = 0
            scores = scores[:, keep_class_mask.bool()]
            num_classes = 18

        num_queries = pred_masks.shape[0]
        labels = (
            torch.arange(num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            test_topk_per_image, sorted=False
        )

        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        pred_masks = pred_masks[topk_indices]

        masks = pred_masks > 0.0
        mask_scores_per_image = (
            pred_masks.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)

        result_3d = {
            "pred_classes": labels_per_image + 1,
            "pred_masks": masks.flatten(1).permute(1, 0),
            "pred_scores": scores_per_image * mask_scores_per_image,
        }
        return result_3d

    def inference_scannet_ghost_semantic(self, mask_cls, mask_pred):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        if self.cfg.MODEL.OPEN_VOCAB:
            mask_cls = mask_cls[..., :-1]
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[:, :-1]

        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qn->cn", mask_cls, mask_pred).max(0)[1]
        return semseg.reshape(-1)

def check_for_nans_in_dict(outputs_dict, prefix=""):
    """Recursively check a (possibly nested) outputs dict for NaN tensors.

    Logs the path of any tensor containing NaN values. Returns ``True`` if any
    NaN was found. This is safe to call in distributed training: it only logs,
    it does not raise or open a debugger.
    """
    found_nan = False
    if not isinstance(outputs_dict, dict):
        return found_nan

    for k, v in outputs_dict.items():
        current_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            found_nan = check_for_nans_in_dict(v, current_key) or found_nan
        elif isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                logger.warning("NaN found in tensor %s, shape=%s", current_key, tuple(v.shape))
                found_nan = True
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    found_nan = (
                        check_for_nans_in_dict(item, f"{current_key}[{i}]") or found_nan
                    )
                elif isinstance(item, torch.Tensor) and torch.isnan(item).any():
                    logger.warning(
                        "NaN found in tensor %s[%d], shape=%s",
                        current_key,
                        i,
                        tuple(item.shape),
                    )
                    found_nan = True
    return found_nan

