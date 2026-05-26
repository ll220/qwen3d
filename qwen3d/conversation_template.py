TEXT_INPUT_GEN_3D = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|pointcloud_pad|><|vision_end|>{target_sentence}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

TEXT_INPUT_TRAIN_3D = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|pointcloud_pad|><|vision_end|>{target_sentence}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{answer}<|im_end|>\n"
)

TEXT_INPUT_GEN_2D = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{target_sentence}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

TEXT_INPUT_TRAIN_2D = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{target_sentence}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{answer}<|im_end|>\n"
)

TEXT_INPUT_TRAIN = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{target_sentence}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{answer}<|im_end|>\n"
)

TEXT_INPUT_GEN = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{target_sentence}<|im_end|>\n"
    "<|im_start|>assistant\n"
)