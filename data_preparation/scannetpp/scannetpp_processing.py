from pathlib import Path
import numpy as np
import pandas as pd
from fire import Fire
from natsort import natsorted
from collections import OrderedDict
import ipdb

from data_preparation.base_preprocessing import BasePreprocessing, load_ply

st = ipdb.set_trace

def filter_map_classes(mapping, count_thresh, count_type, mapping_type):
    # mapping = mapping[mapping[count_type] >= count_thresh]
    if mapping_type == "semantic":
        map_key = "semantic_map_to"
    elif mapping_type == "instance":
        map_key = "instance_map_to"
    else:
        raise NotImplementedError
    # create a dict with classes to be mapped
    # classes that don't have mapping are entered as x->x
    # otherwise x->y
    map_dict = OrderedDict()

    for i in range(mapping.shape[0]):
        row = mapping.iloc[i]
        class_name = row["class"]
        map_target = row[map_key]

        # map to None or some other label -> don't add this class to the label list
        try:
            if len(map_target) > 0:
                # map to None -> don't use this class
                if map_target == "None":
                    pass
                else:
                    # map to something else -> use this class
                    map_dict[class_name] = map_target
        except TypeError:
            # nan values -> no mapping, keep label as is
            if class_name not in map_dict:
                map_dict[class_name] = class_name

    return map_dict


class ScannetPPPreprocessing(BasePreprocessing):
    def __init__(
            self,
            data_dir: str = "./data/raw/scannet/scannet",
            save_dir: str = "./data/processed/scannet",
            modes: tuple = ("train", "validation"),
            n_jobs: int = -1,
    ):
        # n_jobs = 1
        super().__init__(data_dir, save_dir, modes, n_jobs)

        for mode in self.modes:
            trainval_split_dir = Path("splits/scannetpp_splits")
            scannet_special_mode = "val" if mode == "validation" else mode
            with open(
                    trainval_split_dir / (f"nvs_sem_{scannet_special_mode}.txt")
            ) as f:
                # -1 because the last one is always empty
                split_file = f.read().split("\n")[:-1]

            # scans_folder = "scans_test" if mode == "test" else "scans"
            filepaths = []
            for scene in split_file:
                filepaths.append(
                    self.data_dir / "data" / scene / "scans" / "mesh_aligned_0.05.ply"
                )
            self.files[mode] = natsorted(filepaths)

        # Parsing label information and mapping
        # segment_class_names = np.loadtxt(
        #     self.data_dir / "metadata" / "semantic_benchmark" / "top100.txt",
        #     dtype=str,
        #     delimiter=".",  # dummy delimiter to replace " "
        # )
        
        # loading all class names here because locate-3d's snpp assumes all classes
        with open(self.data_dir / "metadata" / "semantic_classes.txt", "r") as f:
            segment_class_names = np.array(
                [line.rstrip("\n") for line in f if line.strip()],
                dtype=str,
            )
        print("Num classes in segment class list:", len(segment_class_names))

        label_mapping = pd.read_csv(
            self.data_dir / "metadata" / "semantic_benchmark" / "map_benchmark.csv"
        )
        self.label_mapping = filter_map_classes(
            label_mapping, count_thresh=0, count_type="count", mapping_type="semantic"
        )
        self.class2idx = {
            class_name: idx for (idx, class_name) in enumerate(segment_class_names)
        }

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        scene = str(filepath).split('/')[-3]
        filebase = {
            "filepath": filepath,
            "scene": scene,
            "sub_scene": "",
            "raw_filepath": str(filepath),
            "file_len": -1,
        }
        # reading both files and checking that they are fitting
        coords, features, _ = load_ply(filepath)
        file_len = len(coords)
        filebase["file_len"] = file_len
        points = np.hstack((coords, features, np.ones_like(coords)))

        if mode in ["train", "validation"]:
            # getting scene information
            scene_path = Path(filepath).parent
            segs_path = scene_path / "segments.json"
            anno_path = scene_path / "segments_anno.json"

            segments = self._read_json(segs_path)
            segments = np.array(segments["segIndices"])

            # add segment id as additional feature
            segment_ids = np.unique(segments, return_inverse=True)[1]
            points = np.hstack((points, segment_ids[..., None]))

            anno = self._read_json(anno_path)

            # adding instance label
            ignore_index = -1
            labels = np.zeros((segments.shape[0], 3))
            empty_instance_label = np.full(labels.shape, -1)
            label_used = np.zeros(segments.shape[0], dtype=np.int16)
            instance_size = np.ones((segments.shape[0], 3), dtype=np.int16) * np.inf

            # labels = np.hstack((labels, empty_instance_label))
            for instance in anno["segGroups"]:
                label = instance['label']
                label = self.label_mapping.get(label, None)
                label_index = self.class2idx.get(label, ignore_index)

                if label_index == ignore_index:
                    continue

                segments_occupied = np.array(instance["segments"])
                occupied_indices = np.isin(segments, segments_occupied)
                additional_mask = label_used < 3
                # if (~additional_mask).sum() > 0:
                #     st()
                occupied_indices = occupied_indices & additional_mask
                size = occupied_indices.sum()
                if size == 0:
                    continue

                label_position = label_used[occupied_indices]
                labels[occupied_indices, label_position] = label_index
                empty_instance_label[occupied_indices, label_position] = instance["objectId"]
                instance_size[occupied_indices, label_position] = size
                label_used[occupied_indices] += 1

            mask = label_used > 1
            if mask.sum() > 0:
                major_label_position = np.argmin(instance_size[mask], axis=1)

                major_semantic_label = labels[mask, major_label_position]
                labels[mask, major_label_position] = labels[:, 0][mask]
                labels[:, 0][mask] = major_semantic_label

                major_instance_label = empty_instance_label[mask, major_label_position]
                empty_instance_label[mask, major_label_position] = empty_instance_label[:, 0][mask]
                empty_instance_label[:, 0][mask] = major_instance_label

            labels = np.hstack((labels[:, :1], empty_instance_label[:, :1]))

            points = np.hstack((points, labels))

            gt_data = points[:, -2] * 1000 + points[:, -1] + 1
        else:
            assert False
            # segments_test = "../../data/raw/scannet_test_segments"
            # segment_indexes_filepath = filepath.name.replace(".ply", ".0.010000.segs.json")
            # segments = self._read_json(f"{segments_test}/{segment_indexes_filepath}")
            # segments = np.array(segments["segIndices"])
            # # add segment id as additional feature
            # segment_ids = np.unique(segments, return_inverse=True)[1]
            # points = np.hstack((points, segment_ids[..., None]))

        processed_filepath = self.save_dir / mode / f"{scene}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        if mode == "test":
            return filebase

        processed_gt_filepath = self.save_dir / "instance_gt" / mode / f"{scene}.txt"
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        filebase["color_mean"] = [
            float((features[:, 0] / 255).mean()),
            float((features[:, 1] / 255).mean()),
            float((features[:, 2] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((features[:, 0] / 255) ** 2).mean()),
            float(((features[:, 1] / 255) ** 2).mean()),
            float(((features[:, 2] / 255) ** 2).mean()),
        ]
        return filebase


if __name__ == "__main__":
    Fire(ScannetPPPreprocessing)
    
    
# python data_preparation/scannetpp/scannetpp_processing.py preprocess \
#     --data_dir /data/group_data/katefgroup-ssd/datasets/scannetpp_full \
#       --save_dir  /data/group_data/katefgroup/language_grounding/mask3d_processed/scannetpp \
#       --modes ["validation",]