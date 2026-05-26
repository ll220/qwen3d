"""
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
"""

import argparse
import time
from pathlib import Path
import urllib.request
from urllib.request import urlretrieve
import urllib.error
import yaml
from munch import Munch
from tqdm import tqdm
import json
import os
import zipfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from scene_release import ScannetppScene_Release


# ------------------------
# Utils
# ------------------------

def read_txt_list(path):
    with open(path) as f:
        return f.read().splitlines()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)
    return Munch.fromDict(y)


def check_remote_file_exists(url):
    request = urllib.request.Request(url)
    request.get_method = lambda: "HEAD"
    try:
        urllib.request.urlopen(request)
        return True
    except urllib.error.HTTPError:
        return False


# ------------------------
# Download helpers
# ------------------------

def urlretrieve_multi_trials(url, filename, max_trials=5):
    for i in range(max_trials):
        try:
            urlretrieve(url, filename)
            time.sleep(0.2)
            return True

        except urllib.error.ContentTooShortError as e:
            print("ERROR: Incomplete download, retrying...")
            if Path(filename).exists():
                os.remove(filename)
            time.sleep(0.5)
            if i == max_trials - 1:
                raise e

        except urllib.error.HTTPError as e:
            print(f"ERROR {e.code} when accessing {url}")
            raise e

    return False


def download_file(url, filename, verbose=True, make_parent=False):
    if make_parent:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"{url} ==> {filename}")
    return urlretrieve_multi_trials(url, filename)


def check_download_file(cfg, url_template, remote_path, local_path, dry_run):
    remote_path = str(remote_path).replace("\\", "/")
    url = url_template.replace("TOKEN", cfg.token).replace("FILEPATH", remote_path)

    if dry_run:
        status = check_remote_file_exists(url)
        print("Remote file exists:" if status else "Remote file missing:", url)
        return status

    if local_path.is_file() or local_path.is_dir():
        if cfg.verbose:
            print("Exists, skipping:", local_path)
        return True

    return download_file(url, local_path, verbose=cfg.verbose, make_parent=True)


# ------------------------
# ScanNet++GS (serial)
# ------------------------

def download_scannetpp_gs(cfg, scene_ids):
    print("Downloading ScanNet++GS data...")
    for scene_id in tqdm(scene_ids, desc="scannetpp_gs"):
        src_path = Path("scannetpp_gs") / scene_id / "ckpts" / "point_cloud_30000.ply"
        tgt_path = Path(cfg.scannetpp_gs_dir) / scene_id / "point_cloud_30000.ply"
        check_download_file(cfg, cfg.scannetpp_gs_url, src_path, tgt_path, cfg.dry_run)


# ------------------------
# Worker
# ------------------------

def process_scene(scene_id, cfg_dict, split_lists, download_assets):
    cfg = Munch(cfg_dict)

    src_scene = ScannetppScene_Release(scene_id, data_root="data")
    tgt_scene = ScannetppScene_Release(
        scene_id, data_root=Path(cfg.data_root) / "data"
    )

    split = None
    for s, ids in split_lists.items():
        if scene_id in ids:
            split = s
            break
    if split is None:
        raise RuntimeError(f"Scene {scene_id} not in any split")

    missing = []

    for asset in download_assets:
        if asset in cfg.exclude_assets.get(split, []):
            continue

        if asset in cfg.zipped_assets:
            tgt_path = getattr(tgt_scene, asset)
            if tgt_path.exists():
                continue

            src_zip = getattr(src_scene, asset).with_suffix(".zip")
            tgt_zip = tgt_path.with_suffix(".zip")

            if not check_download_file(cfg, cfg.root_url, src_zip, tgt_zip, cfg.dry_run):
                missing.append(str(tgt_zip))
                continue

            if not cfg.dry_run:
                with zipfile.ZipFile(tgt_zip, "r") as zf:
                    zf.extractall(tgt_zip.parent)
                tgt_zip.unlink()

        else:
            src_path = getattr(src_scene, asset)
            tgt_path = getattr(tgt_scene, asset)

            if not check_download_file(cfg, cfg.root_url, src_path, tgt_path, cfg.dry_run):
                missing.append(str(tgt_path))

    return scene_id, missing


# ------------------------
# Main
# ------------------------

def main(args):
    cfg = load_yaml_munch(args.config_file)

    # ---- token & path prompts (serial, interactive)
    if cfg.get("token", "<YOUR_TOKEN_HERE>") == "<YOUR_TOKEN_HERE>":
        cfg.token = input("Please enter your download token: ").strip()
        if not cfg.token:
            print("No token provided, exiting.")
            return

    if cfg.get("data_root", "<DOWNLOAD_LOCATION_HERE>") == "<DOWNLOAD_LOCATION_HERE>":
        cfg.data_root = input("Download location [./scannetpp_data]: ").strip() or "./scannetpp_data"

    print(f"Downloading to: {cfg.data_root}")

    print(
        "WARNING: Full dataset ~1.5TB.\n"
        "Proceed? (y/n)\n > ", end=""
    )
    if input().strip().lower() != "y":
        print("Exiting.")
        return

    if cfg.dry_run:
        print("Dry run enabled")

    data_root = Path(cfg.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # ---- metadata (serial)
    missing = []
    for path in cfg.meta_files:
        if not check_download_file(cfg, cfg.root_url, path, data_root / path, cfg.dry_run):
            missing.append(str(data_root / path))

    if cfg.metadata_only:
        print("Metadata downloaded, done.")
        return

    # ---- splits
    split_lists = {}
    for split in cfg.splits:
        split_lists[split] = read_txt_list(
            data_root / "splits" / f"{split}.txt"
        )

    # ---- scenes
    if cfg.get("download_scenes"):
        scene_ids = cfg.download_scenes
    else:
        scene_ids = []
        for split in cfg.download_splits:
            scene_ids += read_txt_list(
                data_root / "splits" / f"{split}.txt"
            )

    # ---- 3rd party GS
    if cfg.get("scannetpp_gs_dir"):
        download_scannetpp_gs(cfg, scene_ids)
        print("ScanNet++GS done.")
        return

    # ---- assets
    if cfg.get("download_assets"):
        download_assets = cfg.download_assets
    elif cfg.get("download_options"):
        download_assets = []
        for opt in cfg.download_options:
            for a in cfg.option_assets[opt]:
                if a not in download_assets:
                    download_assets.append(a)
    else:
        download_assets = cfg.default_assets

    print("Assets:", download_assets)
    print("Scenes:", len(scene_ids))

    # ---- multiprocessing
    cfg_dict = dict(cfg)
    num_workers = cfg.get("num_workers", max(1, mp.cpu_count() // 2))
    # num_workers=1
    print(f"Using {num_workers} workers")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_scene,
                scene_id,
                cfg_dict,
                split_lists,
                download_assets,
            ): scene_id
            for scene_id in scene_ids
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="scenes"):
            scene_id = futures[fut]
            try:
                _, miss = fut.result()
                missing.extend(miss)
            except Exception as e:
                print(f"[ERROR] Scene {scene_id} failed:", e)

    if missing:
        print(f"{len(missing)} files missing:")
        for m in missing:
            print(m)
    else:
        print("Download successful!")


# ------------------------
# Entry point
# ------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
