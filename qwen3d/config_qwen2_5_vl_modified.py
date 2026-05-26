from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    AttentionMaskConverter,
    flash_attn_supports_top_left_mask,
    is_flash_attn_available,
    BaseModelOutputWithPast, 
    ModelOutput,
    ROPE_INIT_FUNCTIONS,
    PreTrainedModel,
    add_start_docstrings, 
    add_start_docstrings_to_model_forward, 
    logging, 
    replace_return_docstrings,
    Qwen2_5_VLConfig, 
    Qwen2_5_VLVisionConfig)

class Qwen2_5_3D_VLConfig(Qwen2_5_VLConfig):
    model_type = "qwen2_5_3d_vl"

    def __init__(self, 
                 pointcloud_token_id=160000,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.pointcloud_token_id = pointcloud_token_id