# Copyright (c) Meta Platforms, Inc. and affiliates.
# inpaint_depth adapted from ODIN (https://github.com/ayushjain1144/odin)

import cv2
import numpy as np


def inpaint_depth(depth: np.ndarray) -> np.ndarray:
    """Fill zero-valued depth pixels using OpenCV TELEA inpainting.

    Args:
        depth: HxW depth array (float32 or compatible).

    Returns:
        Depth array with holes (depth == 0) filled in-place style on a copy.
    """
    depth = np.asarray(depth, dtype=np.float32)
    mask = (depth == 0).astype(np.uint8)
    depth_inpaint = cv2.inpaint(depth, mask, 5, cv2.INPAINT_TELEA)
    out = depth.copy()
    out[depth == 0] = depth_inpaint[depth == 0]
    return out
