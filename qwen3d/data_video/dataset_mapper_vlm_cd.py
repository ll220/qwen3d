"""Stream ``DR8803VL/vlm-change-detection-4k`` examples from the Hub into Qwen-3D samples.

One sample is one example (room + its changes) and one question (see
``datasets/vlm_cd.py``). The mapper:

1. reads the example's two ``walkthrough_frames`` row groups from the Hub (RGB, depth,
   ``c2w`` and per-frame change visibility for both states; the instance maps are never
   downloaded), retrying on network errors;
2. picks ``cfg.VLMCD.FRAMES_PER_STATE`` frames per state (stratified random in training,
   evenly spaced at eval), by default the same camera views in both states;
3. shifts the **after** cameras by ``cfg.VLMCD.AFTER_OFFSET`` metres along +x, so the two
   states back-project into two disjoint copies of the room and never share a voxel when
   Qwen-3D pools all frames. With ``VLMCD.STATE_ENCODING time`` (default) the model then
   removes the shift and feeds each state as its own ``<|vision_start|>`` block at the
   room's true coordinates, the blocks told apart by their RoPE time position (the sample
   carries ``view_states`` and ``state_shift`` for this). With ``offset`` the shifted
   copies go to the LLM as one block;
4. optionally upsamples the 256 px frames to ``cfg.VLMCD.IMAGE_SIZE`` (more ViT tokens per
   frame), adjusting the intrinsics;
5. builds the answer from the changes visible in the chosen frames, back-projects depth
   with the dataset intrinsics and runs the Qwen image processor, like
   ``ScannetDatasetMapper`` does for ScanQA.

The output is a ``generate_only`` sample: the model trains with the language-modelling
loss on the answer, and the mask decoder / ghost points are skipped
(``skip_ghost_points``).
"""

import hashlib
import logging
import os
import random
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from .data_utils import get_multiview_xyz, qwen_preprocess_frames
from .datasets.vlm_cd import ACTIONS, answer, dataset_task, load_index, question

logger = logging.getLogger(__name__)

FRAME_COLUMNS = ["example_id", "frame_index", "height", "width", "rgb_before", "rgb_after",
                 "depth_before", "depth_after", "c2w", "change_visibility"]
STATES = ("before", "after")


def sample_frame_ids(num_frames, k, is_train, rng):
    """Choose ``k`` frame indices out of ``num_frames``.

    Parameters
    ----------
    num_frames : int
        Frames in the walkthrough.
    k : int
        Frames wanted; all frames when ``k <= 0`` or ``k >= num_frames``.
    is_train : bool
        Stratified random (one random frame per equal chunk) when True, evenly spaced
        otherwise.
    rng : random.Random
        Random source for training.

    Returns
    -------
    list of int
        Sorted frame indices.
    """
    if k <= 0 or k >= num_frames:
        return list(range(num_frames))
    if not is_train:
        return np.linspace(0, num_frames - 1, k).round().astype(int).tolist()
    chunks = np.array_split(np.arange(num_frames), k)
    return sorted(int(rng.choice(list(c))) for c in chunks)


def visible_changes(changes, visibility, before_ids, after_ids, min_pixels):
    """Changes seen in the chosen frames.

    A removed object must be seen before, an added one after, and a moved one in either
    state (its old or new place), with at least ``min_pixels`` pixels in one frame.

    Parameters
    ----------
    changes : list of dict
        Index change records (``object``, ``action``, ...).
    visibility : dict of int to list of dict
        ``change_visibility`` per frame index.
    before_ids, after_ids : list of int
        Frames used for each state.
    min_pixels : int
        Pixel threshold at the native resolution.

    Returns
    -------
    list of dict
        Copies of the visible subset of ``changes``, in order, with ``seen_before`` and
        ``seen_after`` added (a moved object's distance is only asked for when both hold).
    """
    seen_before, seen_after = set(), set()
    for i in before_ids:
        for v in visibility.get(i) or []:
            if (v.get("pixels_before") or 0) >= min_pixels:
                seen_before.add(v["object"])
    for i in after_ids:
        for v in visibility.get(i) or []:
            if (v.get("pixels_after") or 0) >= min_pixels:
                seen_after.add(v["object"])
    out = []
    for c in changes:
        o = c["object"]
        if ((c["action"] == "remove" and o in seen_before)
                or (c["action"] == "add" and o in seen_after)
                or (c["action"] == "move" and (o in seen_before or o in seen_after))):
            out.append({**c, "seen_before": o in seen_before, "seen_after": o in seen_after})
    return out


class VlmCdDatasetMapper:
    """Map ``vlmcd_<task>_<split>`` dataset dicts to Qwen-3D samples.

    Parameters
    ----------
    cfg : detectron2.config.CfgNode
        Uses ``cfg.VLMCD.*``, ``cfg.QWEN_MODEL``, ``cfg.INPUT.*`` and the multiview xyz
        settings.
    is_train : bool
    dataset_name : str
        ``vlmcd_<task>_<split>``.
    """

    def __init__(self, cfg, is_train, dataset_name):
        self.cfg = cfg
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.task = dataset_task(dataset_name)
        self.vc = cfg.VLMCD
        self.index = load_index()
        self.examples = self.index["examples"]
        self._fs = None
        self._fs_pid = None

        from transformers import AutoProcessor
        self._qwen_image_processor = AutoProcessor.from_pretrained(
            cfg.QWEN_MODEL, min_pixels=cfg.INPUT.MIN_PIXEL, max_pixels=cfg.INPUT.MAX_PIXEL,
        ).image_processor

    # ------------------------------------------------------------------ reading
    def fs(self):
        """One ``HfFileSystem`` per process (dataloader workers must not share one).

        Returns
        -------
        huggingface_hub.HfFileSystem
        """
        if self._fs is None or self._fs_pid != os.getpid():
            from huggingface_hub import HfFileSystem
            self._fs = HfFileSystem()
            self._fs_pid = os.getpid()
        return self._fs

    def read_example_frames(self, example_id, rec):
        """Read all frames of one example from the Hub (or ``VLMCD.LOCAL_DIR``).

        Parameters
        ----------
        example_id : str
        rec : dict
            The example's index record (``frames.path``, ``frames.row_groups``).

        Returns
        -------
        pyarrow.Table
            The example's rows of ``FRAME_COLUMNS``, sorted by ``frame_index``.
        """
        loc = rec["frames"]
        last = None
        for attempt in range(self.vc.RETRIES):
            try:
                if self.vc.LOCAL_DIR:
                    # A local copy made with `hf download ... --local-dir LOCAL_DIR`.
                    path = os.path.join(self.vc.LOCAL_DIR, "data", loc["path"].split("/data/", 1)[1])
                    f = open(path, "rb")
                else:
                    f = self.fs().open(loc["path"], block_size=8 * 2**20)
                with f:
                    table = pq.ParquetFile(f).read_row_groups(loc["row_groups"], columns=FRAME_COLUMNS)
                ids = table["example_id"].to_numpy(zero_copy_only=False)
                table = table.filter(ids == example_id)
                if table.num_rows == 0:
                    raise KeyError(f"{example_id} not in {loc}")
                return table.sort_by("frame_index")
            except KeyError:
                raise
            except Exception as e:  # network errors, truncated reads
                last = e
                wait = 2 ** attempt + random.random()
                logger.warning(f"vlmcd: reading {example_id} failed ({e!r}); retry in {wait:.1f} s")
                self._fs = None
                time.sleep(wait)
        raise RuntimeError(f"vlmcd: could not read frames of {example_id}") from last

    def load_frames(self, example_id, rec, before_ids, after_ids):
        """Decode the chosen frames, from the eval cache when possible.

        Parameters
        ----------
        example_id : str
        rec : dict
            Index record.
        before_ids, after_ids : list of int

        Returns
        -------
        dict
            ``rgb_<state>`` uint8 ``(V, H, W, 3)``, ``depth_<state>`` uint16 ``(V, H, W)``
            in mm, ``c2w_<state>`` float32 ``(V, 4, 4)``, and ``visibility`` (frame index
            to ``change_visibility``).
        """
        cache = None
        if self.vc.CACHE_DIR and not self.is_train:
            key = hashlib.md5(f"{before_ids}|{after_ids}".encode()).hexdigest()[:10]
            cache = os.path.join(self.vc.CACHE_DIR, f"{example_id}_{key}.npz")
            if os.path.exists(cache):
                with np.load(cache, allow_pickle=True) as z:
                    out = {k: z[k] for k in z.files if k != "visibility"}
                    out["visibility"] = z["visibility"].item()
                return out

        table = self.read_example_frames(example_id, rec)
        fidx = table["frame_index"].to_pylist()
        row = {f: r for r, f in enumerate(fidx)}
        h, w = table["height"][0].as_py(), table["width"][0].as_py()
        c2w = np.asarray(table["c2w"].to_pylist(), dtype=np.float32).reshape(-1, 4, 4)
        vis_col = table["change_visibility"].to_pylist()
        out = {"visibility": {f: vis_col[row[f]] for f in set(before_ids) | set(after_ids)}}
        for state, ids in (("before", before_ids), ("after", after_ids)):
            rows = [row[i] for i in ids]
            rgb_col, depth_col = table[f"rgb_{state}"], table[f"depth_{state}"]
            out[f"rgb_{state}"] = np.stack([
                np.frombuffer(rgb_col[r].as_py(), np.uint8).reshape(h, w, 3) for r in rows])
            out[f"depth_{state}"] = np.stack([
                np.frombuffer(depth_col[r].as_py(), np.uint16).reshape(h, w) for r in rows])
            out[f"c2w_{state}"] = c2w[rows]
        if cache:
            os.makedirs(self.vc.CACHE_DIR, exist_ok=True)
            tmp = cache + f".{os.getpid()}.tmp.npz"
            np.savez(tmp, **{k: v for k, v in out.items() if k != "visibility"},
                     visibility=np.array(out["visibility"], dtype=object))
            os.replace(tmp, cache)
        return out

    # ---------------------------------------------------------------- geometry
    def intrinsics(self, rec, scale):
        """4x4 intrinsics in Qwen-3D's convention, at the (possibly upsampled) size.

        The dataset uses pixel centres at ``u + 0.5``; ``backproject.unproject`` uses
        ``u``, so ``cx`` and ``cy`` move half a pixel.

        Parameters
        ----------
        rec : dict
            Index record with ``intrinsics`` (``fx``, ``fy``, ``cx``, ``cy``).
        scale : float
            Output size over native size.

        Returns
        -------
        numpy.ndarray
            float32 ``(4, 4)``.
        """
        k = rec["intrinsics"]
        out = np.eye(4, dtype=np.float32)
        out[0, 0] = k["fx"] * scale
        out[1, 1] = k["fy"] * scale
        out[0, 2] = k["cx"] * scale - 0.5
        out[1, 2] = k["cy"] * scale - 0.5
        return out

    def resize(self, rgb, depth, size):
        """Resize one frame: bilinear colour, nearest depth.

        Parameters
        ----------
        rgb : numpy.ndarray
            uint8 ``(H, W, 3)``.
        depth : numpy.ndarray
            float32 ``(H, W)`` in metres.
        size : int
            Output side in pixels (frames are square).

        Returns
        -------
        tuple of numpy.ndarray
        """
        if rgb.shape[0] == size and rgb.shape[1] == size:
            return rgb, depth
        rgb = np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))
        depth = np.asarray(Image.fromarray(depth).resize((size, size), Image.NEAREST))
        return rgb, depth

    # -------------------------------------------------------------------- call
    def __call__(self, dataset_dict):
        """Build one sample.

        Parameters
        ----------
        dataset_dict : dict
            ``{"example_id", "task", "split", "image_id"}``.

        Returns
        -------
        dict or None
            The Qwen-3D sample. ``None`` in training when the frames could not be read,
            so detectron2's ``MapDataset`` retries with another example.
        """
        example_id = dataset_dict["example_id"]
        rec = self.examples[example_id]
        n = rec["num_frames"]
        k = int(self.vc.FRAMES_PER_STATE)
        rng = random.Random() if self.is_train else random.Random(0)
        before_ids = sample_frame_ids(n, k, self.is_train, rng)
        after_ids = before_ids if self.vc.SAME_FRAMES else sample_frame_ids(n, k, self.is_train, rng)

        try:
            fr = self.load_frames(example_id, rec, before_ids, after_ids)
        except Exception:
            if self.is_train:
                logger.exception(f"vlmcd: skipping {example_id}")
                return None
            raise

        changes = rec["changes"]
        if self.vc.ONLY_VISIBLE_CHANGES:
            changes = visible_changes(changes, fr["visibility"], before_ids, after_ids,
                                      self.vc.MIN_VISIBLE_PIXELS)
        prompt = question(self.task, self.vc.STATE_ENCODING, self.vc.ANSWER_FORMAT)
        target = answer(self.task, changes, self.vc.ANSWER_ORDER, self.vc.ANSWER_FORMAT)

        native = fr["rgb_before"].shape[1]
        size = int(self.vc.IMAGE_SIZE) if self.vc.IMAGE_SIZE > 0 else native
        K = torch.from_numpy(self.intrinsics(rec, size / native))
        shift = np.eye(4, dtype=np.float32)
        shift[0, 3] = self.vc.AFTER_OFFSET

        images, depths, poses, intrinsics = [], [], [], []
        for state in STATES:
            for v in range(len(fr[f"rgb_{state}"])):
                depth = fr[f"depth_{state}"][v].astype(np.float32) / 1000.0
                if self.cfg.IGNORE_DEPTH_MAX != -1:
                    depth[depth > self.cfg.IGNORE_DEPTH_MAX] = 0.0
                rgb, depth = self.resize(fr[f"rgb_{state}"][v], depth, size)
                pose = fr[f"c2w_{state}"][v]
                if state == "after":
                    pose = shift @ pose
                images.append(torch.as_tensor(np.ascontiguousarray(rgb.transpose(2, 0, 1))))
                depths.append(torch.as_tensor(np.ascontiguousarray(depth)))
                poses.append(torch.from_numpy(np.ascontiguousarray(pose)).float())
                intrinsics.append(K.clone())

        num_views = len(images)
        h, w = images[0].shape[-2:]
        multi_scale_xyz, _, original_xyz, new_h, new_w = get_multiview_xyz(
            shape=(num_views, h, w),
            size_divisibility=self.cfg.INPUT.SIZE_DIVISIBILITY,
            depths=depths,
            poses=poses,
            intrinsics=intrinsics,
            is_train=self.is_train,
            augment_3d=self.cfg.INPUT.AUGMENT_3D,
            interpolation_method=self.cfg.MODEL.INTERPOLATION_METHOD,
            mask_valid=self.cfg.MASK_VALID,
            mean_center=self.cfg.MEAN_CENTER,
            do_rot_scale=self.cfg.DO_ROT_SCALE,
            scannet_pc=None,
            align_matrix=None,
            vil3d=self.cfg.VIL3D,
            scales=self.cfg.MULTIVIEW_XYZ_SCALES,
            min_pixel=self.cfg.INPUT.MIN_PIXEL,
            max_pixel=self.cfg.INPUT.MAX_PIXEL,
        )
        pv_list, grid_list = qwen_preprocess_frames(self._qwen_image_processor, images)

        # STATE_ENCODING time: tell the model which views belong to which scan and how far
        # each scan was shifted, so it can pool them apart, undo the shift and give each
        # scan its own <|vision_start|> block (Qwen3D.group_voxels_by_state).
        state_info = {}
        if self.vc.STATE_ENCODING == "time":
            state_info = {
                "view_states": [i for i, s in enumerate(STATES) for _ in fr[f"rgb_{s}"]],
                "state_shift": [[0.0, 0.0, 0.0], [float(self.vc.AFTER_OFFSET), 0.0, 0.0]],
            }

        counts = {a: sum(c["action"] == a for c in changes) for a in ACTIONS}
        lang = {
            "text_caption": prompt,
            "answers": [target],
            "target_id": None,
            "anchor_ids": [],
            "target_name": None,
            "anchors_names": [],
            "do_generate": True,
            "example_id": example_id,
            "task": self.task,
            "visible_changes": [[c["action"], c["category"]] for c in changes],
            "counts": counts,
        }
        return {
            "dataset_name": self.dataset_name,
            "image_id": example_id,
            "file_name": example_id,
            "file_names": [f"{example_id}/{s}/{i}" for s in STATES
                           for i in (before_ids if s == "before" else after_ids)],
            "frame_ids": {"before": before_ids, "after": after_ids},
            "images": images,
            "depths": depths,
            "poses": poses,
            "intrinsics": intrinsics,
            "multi_scale_xyz": multi_scale_xyz,
            "original_xyz": original_xyz,
            "new_h": new_h,
            "new_w": new_w,
            "height": h,
            "width": w,
            "new_image_shape": (h, w),
            "qwen_pixel_values": pv_list,
            "qwen_grid_thw": grid_list,
            "qwen_fcs": None,
            "qwen_feature_paths": None,
            "decoder_3d": True,
            "actual_decoder_3d": True,
            "skip_ghost_points": True,
            "full_scene_dataset": True,
            "generate_only": True,
            "do_generate": True,
            "sr3d_data": [lang],
            "num_classes": 1,
            "all_classes": {0: "object"},
            "valid_class_ids": np.arange(1),
            # collate_fn (camera-drop augmentation) settings: off.
            "do_camera_drop": False,
            "camera_drop_prob": 0.0,
            "camera_drop_min_frames_keep": 1,
            "always_keep_first_frame": True,
            "max_frames": num_views,
            "use_ghost": False,
            "pseudo_2d_aug": False,
            **state_info,
        }
