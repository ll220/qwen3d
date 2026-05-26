"""
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import zlib
import numpy as np
import imageio as iio
import lz4.block
from tqdm import tqdm
from munch import Munch

from common.scene_release import ScannetppScene_Release
from common.utils.utils import run_command, load_yaml_munch, read_txt_list


# ------------------------
# Extraction functions
# ------------------------

def extract_rgb(scene):
    scene.iphone_rgb_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"ffmpeg -i {scene.iphone_video_path} "
        f"-start_number 0 -q:v 1 "
        f"{scene.iphone_rgb_dir}/frame_%06d.jpg"
    )
    run_command(cmd, verbose=True)


def extract_masks(scene):
    scene.iphone_video_mask_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"ffmpeg -i {scene.iphone_video_mask_path} "
        f"-pix_fmt gray -start_number 0 "
        f"{scene.iphone_video_mask_dir}/frame_%06d.png"
    )
    run_command(cmd, verbose=True)


def extract_depth(scene):
    height, width = 192, 256
    sample_rate = 1
    scene.iphone_depth_dir.mkdir(parents=True, exist_ok=True)

    # global compression with zlib
    try:
        with open(scene.iphone_depth_path, 'rb') as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in tqdm(
            range(0, depth.shape[0], sample_rate),
            desc=f"decode_depth ({scene.scene_id})",
            leave=False,
        ):
            iio.imwrite(
                f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png",
                (depth[frame_id] * 1000).astype(np.uint16),
            )

    # per-frame compression with lz4 / zlib
    except Exception:
        frame_id = 0
        with open(scene.iphone_depth_path, 'rb') as infile:
            while True:
                size = infile.read(4)
                if len(size) == 0:
                    break

                size = int.from_bytes(size, byteorder='little')

                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                data = infile.read(size)
                try:
                    data = lz4.block.decompress(
                        data,
                        uncompressed_size=height * width * 2,
                    )
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                except Exception:
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = (
                        np.frombuffer(data, dtype=np.float32)
                        .reshape(height, width) * 1000
                    ).astype(np.uint16)

                iio.imwrite(
                    f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png",
                    depth,
                )
                frame_id += 1


# ------------------------
# Worker (must be top-level)
# ------------------------

def process_scene(scene_id, cfg_dict):
    cfg = Munch(cfg_dict)
    scene = ScannetppScene_Release(
        scene_id, data_root=Path(cfg.data_root) / 'data'
    )

    if cfg.extract_rgb:
        extract_rgb(scene)

    if cfg.extract_masks:
        extract_masks(scene)

    if cfg.extract_depth:
        extract_depth(scene)


# ------------------------
# Main
# ------------------------

def main(args):
    cfg = load_yaml_munch(args.config_file)

    # determine scene list
    if cfg.get('scene_list_file'):
        scene_ids = read_txt_list(cfg.scene_list_file)
    elif cfg.get('scene_ids'):
        scene_ids = cfg.scene_ids
    elif cfg.get('splits'):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / 'splits' / f'{split}.txt'
            scene_ids += read_txt_list(split_path)
    else:
        raise ValueError("No scene specification provided")

    # multiprocessing-safe config
    cfg_dict = dict(cfg)

    num_workers = cfg.get('num_workers', max(1, mp.cpu_count() // 2))
    print(f"Processing {len(scene_ids)} scenes with {num_workers} workers")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_scene, scene_id, cfg_dict): scene_id
            for scene_id in scene_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="scene"):
            scene_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Scene {scene_id} failed: {e}")


# ------------------------
# Entry point
# ------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()

    main(args)
