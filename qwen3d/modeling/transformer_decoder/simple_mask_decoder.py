# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging

import ipdb
import torch
from detectron2.config import configurable
from detectron2.utils.registry import Registry
from qwen3d.modeling.backproject.backproject import voxel_map_to_source
from qwen3d.modeling.meta_arch.self_cross_attention_layers import (
    MLP,
    CrossAttentionLayer,
    FFNLayer,
    SelfAttentionLayer,
)
from qwen3d.modeling.transformer_decoder.position_encoding import PositionEmbeddingLearned, PositionEmbeddingSine3D
from qwen3d.utils.misc import nanmax, nanmin
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from einops import rearrange

st = ipdb.set_trace


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


@TRANSFORMER_DECODER_REGISTRY.register()
class SimpleMaskDecoder(nn.Module):
    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        decoder_3d: bool,
        cfg=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # self.num_frames = num_frames
        self.decoder_3d = decoder_3d
        self.hidden_dim = hidden_dim
        self.cfg = cfg

        self.pe_layer, self.pe_layer_2d = self.init_pe()

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers_f_to_q = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.vis_output_cross_attn = nn.ModuleList()
        self.transformer_text_cross_attention_layers = nn.ModuleList()
        self.vis_output_ffn = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # output FFNs
        if self.mask_classification and not self.cfg.MODEL.OPEN_VOCAB:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        if self.cfg.MODEL.OPEN_VOCAB:
            self.max_seq_len = cfg.MODEL.MAX_SEQ_LEN
            self.lang_pos_embed = nn.Embedding(self.max_seq_len, hidden_dim)
            self.class_embed = nn.Linear(hidden_dim, hidden_dim)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES

        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["decoder_3d"] = cfg.MODEL.DECODER_3D
        ret["cfg"] = cfg

        return ret

    def init_pe(self):
        pe_layer_3d, pe_layer_2d = None, None
        N_steps = self.hidden_dim // 2
        if self.cfg.MODEL.DECODER_3D:
            pe_layer_3d = PositionEmbeddingLearned(
                dim=3, num_pos_feats=self.hidden_dim
            )
        if self.cfg.MODEL.DECODER_2D:
            pe_layer_2d = PositionEmbeddingSine3D(
                N_steps, normalize=True, add_temporal=self.cfg.ADD_TEMPORAL
            )
        return pe_layer_3d, pe_layer_2d

    def open_vocab_class_pred(self, decoder_output, text_feats):
        class_embed = self.class_embed(decoder_output)
        query_feats = F.normalize(class_embed, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        output_class = torch.einsum("bqc,sbc->bqs", query_feats / 0.07, text_feats)
        return output_class

    def init_object_queries(
        self, bs
    ):
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        return query_embed, output

    def forward(
        self,
        mask_features,
        shape=None,
        mask=None,
        mask_features_xyz=None,
        segments=None,
        decoder_3d=False,
        actual_decoder_3d=False,
        scannet_all_masks_batched=None,
        max_valid_points=None,
        qwen_3d_text_features=None,
    ):
        c = mask_features.shape[1]
        # max_valid_points = max_valid_points_segments if self.cfg.USE_SEGMENTS else max_valid_points
        voxelize = (
            decoder_3d and self.cfg.INPUT.VOXELIZE
        )
        # treat_force_3d_as_2d = not actual_decoder_3d and self.cfg.TREAT_FORCE_DECODER_3D_AS_2D

        if shape is None:
            assert not decoder_3d
            shape = [mask_features[0].shape[0], 1]
        bs, v = shape
        pe_layer = self.pe_layer if (decoder_3d or self.cfg.PASS_2D_CROSS_VIEW) else self.pe_layer_2d

        if not decoder_3d:
            _, c_m, h_m, w_m = mask_features.shape
            mask_features = mask_features.view(bs, v, c_m, h_m, w_m)
            self.forward_prediction_heads = self.forward_prediction_heads2D
            assert not voxelize
        else:
            if self.cfg.INPUT.VOXELIZE:
                assert mask_features.shape[-1] == 1, mask_features.shape
                mask_features = mask_features[..., 0]
                # voxelize (mask features are already voxelized)
                assert mask_features_xyz.shape[-2] == mask_features.shape[-1]
                if self.cfg.USE_GT_MASKS:
                    mask_features = scatter_mean(
                        mask_features.permute(0, 2, 1), scannet_all_masks_batched.long(), dim=1
                    ).permute(
                        0, 2, 1
                    )
                elif self.cfg.USE_SEGMENTS:
                    mask_features = scatter_mean(
                        mask_features.permute(0, 2, 1), segments, dim=1
                    ).permute(
                        0, 2, 1
                    )  # B, C, N

                self.forward_prediction_heads = self.forward_prediction_heads3D
            else:
                _, c_m, h_m, w_m = mask_features.shape
                mask_features = mask_features.view(bs, v, c_m, h_m, w_m)
                self.forward_prediction_heads = self.forward_prediction_heads2D

        text_feats = None
        text_attn_mask = None
        if self.cfg.MODEL.OPEN_VOCAB:
            assert qwen_3d_text_features.shape[0] == 1, "Qwen3D text features should be of shape [1, S, C]"
            text_feats = qwen_3d_text_features
            text_attn_mask = torch.zeros(
                qwen_3d_text_features.shape[1], device=qwen_3d_text_features.device).unsqueeze(0).bool()
            
            text_feats = text_feats.permute(1, 0, 2)  # S X B X C
            bs = text_feats.shape[1]
            lang_pos_embed = self.lang_pos_embed.weight[:, None].repeat(1, bs, 1)[
                : text_feats.shape[0]
            ]           
            
        if decoder_3d:
            if self.cfg.USE_GT_MASKS:
                mask_features_xyz_segments = scatter_mean(
                    mask_features_xyz, scannet_all_masks_batched, dim=1
                )
            elif self.cfg.USE_SEGMENTS and segments is not None:
                mask_features_xyz_segments = scatter_mean(
                    mask_features_xyz, segments, dim=1
                )
            else:
                mask_features_xyz_segments = mask_features_xyz

        query_embed, output = self.init_object_queries(bs)

        query_pad_mask = None
        predictions_class = []
        predictions_mask = []

        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output,
            mask_features,
            segments=segments,
            text_feats=text_feats,
            scannet_all_masks_batched=scannet_all_masks_batched,
        )

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        if decoder_3d or self.cfg.PASS_2D_CROSS_VIEW:
            mask_features_pos = pe_layer(mask_features_xyz_segments).permute(1, 0, 2)
        else:
            hw_ = mask_features.shape[-2] * mask_features.shape[-1]
            mask_features_pos = pe_layer(
                mask_features, None
            ).flatten(3).view(bs, v, c, hw_).permute(1, 3, 0, 2).flatten(0, 1)
        
        if decoder_3d:
            vis_feats = mask_features.permute(2, 0, 1)
        else:
            vis_feats = rearrange(mask_features, "b v c h w -> (v h w) b c")
            
        vis_lang_feats = torch.cat([
            vis_feats, text_feats
        ])
        vis_lang_pos = torch.cat([
            mask_features_pos, lang_pos_embed
        ])

        for i in range(self.num_layers):
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            attn_mask = torch.cat(
                [
                    attn_mask,
                    torch.zeros(
                        attn_mask.shape[0],
                        attn_mask.shape[1],
                        text_feats.shape[0],
                        device=output.device,
                    ).bool(),
                ],
                2,
            )

            output = self.transformer_cross_attention_layers[i](
                output,
                vis_lang_feats,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=vis_lang_pos,
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=query_pad_mask,
                query_pos=query_embed,
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                segments=segments,
                text_feats=text_feats,
                scannet_all_masks_batched=scannet_all_masks_batched,
            )

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1


        predictions_boxes = None
        if self.cfg.USE_BOX_LOSS and actual_decoder_3d and self.training:
            predictions_boxes = []
            for i in range(len(predictions_mask)):
                masks = predictions_mask[i] > 0

                # remove padded tokens
                for j in range(len(max_valid_points)):
                    masks[j, :, max_valid_points[j]:] = False

                pc = mask_features_xyz_segments

                pc = pc[:, None].repeat(1, masks.shape[1], 1, 1)
                assert pc.shape[2] == masks.shape[2], f"inconsistent pc and mask shapes. pc: {pc.shape}, mask: {mask.shape}"
                pc[torch.where(masks == 0)] = torch.nan


                boxes = torch.cat([
                    nanmin(pc, dim=2)[0], nanmax(pc, dim=2)[0]
                ], 2)

                # if only one point is in the mask, the box will still be too small
                boxes[masks.sum(2) <= 1] = torch.tensor([0, 0, 0, 1e-2, 1e-2, 1e-2], device=boxes.device)
                predictions_boxes.append(boxes)
        out = {
            "text_attn_mask": text_attn_mask,
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            # "sensor_pred_masks": sensor_prediction_masks[-1] if self.cfg.SUPERVISE_SENSOR_MASKS else None,
            'pred_boxes': predictions_boxes[-1] if predictions_boxes is not None else None,
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask, predictions_boxes,
                None
            ),

        }

        # USE_CLASSIFICATION_ONLY_LOSS is not used but useful to keep in here just in case
        if self.cfg.USE_CLASSIFICATION_ONLY_LOSS:
            if not decoder_3d:
                mask_features = rearrange(mask_features, "b v c h w -> b c (v h w)")
            outputs_class = self.open_vocab_class_pred(
                mask_features.permute(0, 2, 1), text_feats
            )
            out["cls_only_logits"] = outputs_class
            if "aux_outputs" in out:
                for aux_out in out["aux_outputs"]:
                    aux_out["cls_only_logits"] = outputs_class

        return out

    def forward_prediction_heads3D(
        self,
        output,
        mask_features,
        segments=None,
        text_feats=None,
        scannet_all_masks_batched=None,
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        outputs_class = self.open_vocab_class_pred(
            decoder_output, text_feats,
        )

        mask_embed = self.mask_embed(decoder_output)

        segment_mask = torch.einsum("bqc,bcn->bqn", mask_embed, mask_features)
        if self.cfg.USE_GT_MASKS:
            output_mask = voxel_map_to_source(
                segment_mask.permute(0, 2, 1), scannet_all_masks_batched
            ).permute(0, 2, 1)
        elif self.cfg.USE_SEGMENTS:
            output_mask = voxel_map_to_source(
                segment_mask.permute(0, 2, 1), segments
            ).permute(0, 2, 1)
        else:
            output_mask = segment_mask
            segment_mask = None

        attn_mask = segment_mask if self.cfg.USE_SEGMENTS else output_mask
        
        attn_mask = (
            attn_mask.sigmoid()
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        if self.cfg.USE_SEGMENTS:
            output_mask = segment_mask

        return outputs_class, output_mask, attn_mask
    
    
    def forward_prediction_heads2D(
        self,
        output,
        mask_features,
        segments=None,
        text_feats=None,
        scannet_all_masks_batched=None
    ):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        outputs_class = self.open_vocab_class_pred(
            decoder_output, text_feats,
        )

        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)
        attn_mask = outputs_mask.flatten(2)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()
        return outputs_class, outputs_mask, attn_mask
    

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_boxes, outputs_lang):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if outputs_boxes is not None and self.cfg.GENERATION:
        #     return [
        #         {"pred_logits": a, "pred_masks": b, "pred_boxes": c, "generation_logits": d}
        #         for a, b, c, d in zip(
        #             outputs_class[:-1], outputs_seg_masks[:-1], outputs_boxes[:-1], outputs_lang[:-1]
        #         )
        #     ]
        if outputs_boxes is not None:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_boxes[:-1])
            ]
        elif self.mask_classification:
            # if self.cfg.GENERATION:
            #     return [
            #         {"pred_logits": a, "pred_masks": b, "generation_logits": c}
            #         for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_lang[:-1])
            #     ]
            # else:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b, in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
