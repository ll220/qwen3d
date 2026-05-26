# DIR="$(dirname "$PWD")"
# export PYTHONPATH=$DIR:$DIR/pretrain:$PWD

import numpy as np
import sys
from PIL import Image
import torch

import ipdb
st = ipdb.set_trace

from fvcore.transforms.transform import (
    HFlipTransform,
    NoOpTransform,
    VFlipTransform,
)
from detectron2.data import transforms as T

class RandomApplyClip(T.Augmentation):
    """
    Randomly apply an augmentation with a given probability.
    """

    def __init__(self, tfm_or_aug, prob=0.5, clip_frame_cnt=1):
        """
        Args:
            tfm_or_aug (Transform, Augmentation): the transform or augmentation
                to be applied. It can either be a `Transform` or `Augmentation`
                instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()
        self.aug = T.augmentation._transform_to_aug(tfm_or_aug)
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob
        self._cnt = 0
        self.clip_frame_cnt = clip_frame_cnt

    def get_transform(self, *args):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        if self.do:
            return self.aug.get_transform(*args)
        else:
            return NoOpTransform()

    def __call__(self, aug_input):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        if self.do:
            return self.aug(aug_input)
        else:
            return NoOpTransform()


class RandomRotationClip(T.Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(self, angle, prob=0.5, expand=True, center=None, interp=None, clip_frame_cnt=1):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self.angle_save = None
        self.center_save = None
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        if self._cnt % self.clip_frame_cnt == 0:
            center = None
            angle = np.random.uniform(self.angle[0], self.angle[1], size=self.clip_frame_cnt)
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
            angle = np.sort(angle)
            if self._rand_range() < self.prob:
                angle = angle[::-1]
            self.angle_save = angle
            self.center_save = center

            self._cnt = 0   # avoiding overflow

        angle = self.angle_save[self._cnt]
        center = self.center_save

        self._cnt += 1

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return T.RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)

class ResizeShortestEdgeClip(T.Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR, clip_frame_cnt=1
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice", "range_by_clip", "choice_by_clip"], sample_style

        self.is_range = ("range" in sample_style)
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            if self.is_range:
                self.size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
            else:
                self.size = np.random.choice(self.short_edge_length)
            self._cnt = 0   # avoiding overflow

            if self.size == 0:
                return NoOpTransform()
        self._cnt += 1

        h, w = image.shape[:2]

        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, scale * w
        else:
            newh, neww = scale * h, self.size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return T.ResizeTransform(h, w, newh, neww, self.interp)
    
class RandomFlipClip(T.Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False, clip_frame_cnt=1):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._cnt = 0

        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        if self.do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()
        
class RandomCropClip(T.Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size, clip_frame_cnt=1):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.
        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        # TODO style of relative_range and absolute_range are not consistent:
        # one takes (h, w) but another takes (min, max)
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())
        self._cnt = 0

    def get_transform(self, image):
        h, w = image.shape[:2]      # 667, 500
        if self._cnt % self.clip_frame_cnt == 0:
            croph, cropw = self.get_crop_size((h, w))
            assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)

            h0 = np.random.randint(h - croph + 1)   # rand(124) -> 5
            w0 = np.random.randint(w - cropw + 1)   # rand(111) -> 634

            h1 = np.random.randint(h0, h - croph + 1)
            w1 = np.random.randint(w0, w - cropw + 1)

            x = np.sort(np.random.rand(self.clip_frame_cnt))

            h = h0 * x + h1 * (1-x)
            w = w0 * x + w1 * (1-x)
            h = np.round(h).astype(int)
            w = np.round(w).astype(int)

            if self._rand_range() < 0.5:
                h = h[::-1]
                w = w[::-1]

            self.hw_save = (h, w)
            self.crop_h_save, self.crop_w_save = croph, cropw
            self._cnt = 0   # avoiding overflow
        _h, _w = self.hw_save[0][self._cnt], self.hw_save[1][self._cnt]
        self._cnt += 1

        return T.CropTransform(_w, _h, self.crop_w_save, self.crop_h_save)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width
        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))
        

def build_pseudo_augmentation(clip_frame_cnt):
    aug_list = []
    
    min_size = (288, 320, 352, 384, 416, 448, 480, 512)
    max_size = 768
    sample_style = "choice_by_clip"
    

    # Crop
    crop_type = "absolute_range"
    crop_size = (384, 600)
    crop_aug = RandomApplyClip(
        T.AugmentationList([
            ResizeShortestEdgeClip([400, 500, 600], 1333, sample_style, clip_frame_cnt=clip_frame_cnt),
            RandomCropClip(crop_type, crop_size, clip_frame_cnt=clip_frame_cnt)
        ]),
        clip_frame_cnt=clip_frame_cnt
    )
    aug_list.append(crop_aug)

    # Resize
    aug_list.append(ResizeShortestEdgeClip(min_size, max_size, sample_style, clip_frame_cnt=clip_frame_cnt))

    # Flip
    aug_list.append(
        # NOTE using RandomFlip modified for the support of flip maintenance
        RandomFlipClip(
            horizontal=True,
            vertical=False,
            clip_frame_cnt=clip_frame_cnt,
        )
    )

    # rotation
    aug_list.append(
        RandomRotationClip(
            [-15, 15], expand=False, center=[(0.4, 0.4), (0.6, 0.6)], clip_frame_cnt=clip_frame_cnt,
        )
    )
    
    return aug_list



def augment_pointmap(pointmap, points_transforms, mask=None):
    """sumary_line
    
    Keyword arguments:
        pointmap -- xyz of shape [V, H, W, 3]
        points_transforms -- list of transforms to apply on the point of length V
    Return: pointmap after applying the transforms of shape [V, H, W, 3]
    """
    # Lucy: band-aid solution so code can run for now, feel free to delete :)
    if len(points_transforms) == 0 and len(pointmap) == 1:
        pointmap_ = pointmap[0]
        pointmap_[mask[0] == 0] = -10.0
        pointmap = [pointmap_]
        return torch.from_numpy(np.stack(pointmap)).unsqueeze(0)

    assert pointmap.shape[0] == len(points_transforms)

    all_pointmaps = []
    for i, point_transform in enumerate(points_transforms):
        for j, t in enumerate(point_transform):
            if isinstance(t, T.ResizeTransform):
                point_transform[j].interp = Image.NEAREST
            if isinstance(t, T.RotationTransform):
                point_transform[j].interp = Image.NEAREST

        if mask is not None:
            padding_mask = mask[i].astype((np.float32))
        else:
            padding_mask = np.ones(pointmap[i].shape[:2])
        padding_mask = point_transform.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        pointmap_ = T.apply_transform_gens(point_transform, T.AugInput(pointmap[i]))[0].image
        pointmap_[padding_mask] = -10.0
        all_pointmaps.append(pointmap_)

    point_map = torch.from_numpy(np.stack(all_pointmaps))

    return point_map
    