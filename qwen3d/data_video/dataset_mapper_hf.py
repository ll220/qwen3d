from detectron2.config import configurable
import torch
import numpy as np
import math
from .data_utils import round_by_factor, floor_by_factor, ceil_by_factor
from PIL import Image


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    image_size = cfg.INPUT.IMAGE_SIZE_2D
    # min_scale = cfg.INPUT.MIN_SCALE
    # max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if is_train:
        if cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
        augmentation.extend(
            [
                # T.ResizeScale(
                #     min_scale=min_scale,
                #     max_scale=max_scale,
                #     target_height=image_size,
                #     target_width=image_size,
                # ),
                NoOpTransform(),
                T.FixedSizeCrop(crop_size=(image_size, image_size)),
            ]
        )
    else:
        # min_size = cfg.INPUT.MIN_SIZE_TEST_2D
        # max_size = cfg.INPUT.MAX_SIZE_TEST_2D
        # sample_style = "choice"
        # augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        return None

    return augmentation


def format_mcq_prompt(question: str, options: list[str]) -> str:
    """
    Format a multiple-choice question and options into the target prompt style.

    Example output:
    'Which of the 3 objects is the smallest?
    A. The object on the right is the smallest object.
    B. The object on the left is the smallest object.
    C. The object in the middle is the smallest object.
    Please answer directly with only the letter of the correct option and nothing else.'
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [question.strip()]
    for i, opt in enumerate(options):
        lines.append(f"{letters[i]}. {opt.strip()}")
    lines.append("Please answer directly with only the letter of the correct option and nothing else.")
    return "\n".join(lines)

class HFDatasetTextMapper:
    def __init__(self, cfg, dataset_name):
        self.cfg = cfg
        self.dataset_name = dataset_name

    def __call__(self, dataset_dict):
        output_dict = {}
        output_dict["dataset_name"] = self.dataset_name
        if self.dataset_name == "alpaca_text_bench":
            output_dict["text_caption"] = dataset_dict["instruction"] + "\n" + dataset_dict["input"]
            output_dict["answer"] = dataset_dict["output"]
        elif self.dataset_name == "mmlupro_text_bench":
            output_dict["text_caption"] = format_mcq_prompt(question=dataset_dict["question"], options=dataset_dict["options"])
            output_dict["answer"] = dataset_dict["answer"]

        output_dict['generate_only'] = True
        output_dict['do_generate'] = True
        output_dict['text_only'] = True
        output_dict["decoder_3d"] = False
        output_dict["actual_decoder_3d"] = False

        output_dict["width"] = 0
        output_dict["height"] = 0
        output_dict["do_camera_drop"] = False
        output_dict["camera_drop_prob"] = 0.0
        output_dict["camera_drop_min_frames_keep"] = 1
        output_dict["always_keep_first_frame"] = True
        output_dict["max_frames"] = 1
        output_dict["use_ghost"] = False

        return output_dict

        
# TODO: add transforms if needed for training setup
class HFDatasetMapper:
    # @configurable
    def __init__(self, cfg, dataset_name, features, is_train=False):
        assert "image" in features.keys() 
        assert "question" in features.keys()
        assert "answer" in features.keys()
        self.cfg = cfg
        self.features = features
        self.is_train = is_train
        self.dataset_name = dataset_name
        

    def __call__(self, dataset_dict):
        output_dict = {}
        output_dict["decoder_3d"] = False
        output_dict["actual_decoder_3d"] = False

        image = dataset_dict["image"]

        if isinstance(image, str):
            assert "image_file" in dataset_dict.keys(), "If image is given as path, 'image_file' key must be present."
            image = Image.open(dataset_dict["image_file"]).convert("RGB")

        image = [torch.from_numpy(np.array(image)).permute(2,0,1).contiguous()]  # (C,H,W)
        output_dict["images"] = image
        output_dict["dataset_name"] = self.dataset_name
        output_dict["answer"] = dataset_dict["answer"]
        output_dict["text_caption"] = dataset_dict["question"]
        # output_dict["sr3d_data"][0]["answers"] = [dataset_dict["answer"]]
        # output_dict["sr3d_data"][0]["text_captions"] = dataset_dict["question"]

        h, w = image[0].shape[-2:]
        output_dict["width"] = w
        output_dict["height"] = h
        scales = self.cfg.MULTIVIEW_XYZ_SCALES
        max_pixel = self.cfg.INPUT.MAX_PIXEL
        min_pixel = self.cfg.INPUT.MIN_PIXEL
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
        output_dict["new_h"] = ds_h 
        output_dict["new_w"] = ds_w
        output_dict['generate_only'] = True
        output_dict["do_camera_drop"] = False
        output_dict["camera_drop_prob"] = 0.0
        output_dict["camera_drop_min_frames_keep"] = 1
        output_dict["always_keep_first_frame"] = True
        output_dict["max_frames"] = 1
        output_dict["use_ghost"] = False
        output_dict["dataset_name"] = self.dataset_name
        output_dict['pseudo_2d_aug'] = self.cfg.PSEUDO_2D_AUG
        dataset_dict['do_generate'] = True
        dataset_dict['is_train'] = self.is_train
        return output_dict