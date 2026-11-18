"""Change-detection QA on ``DR8803VL/vlm-change-detection-4k`` (MolmoSpaces point cloud difference).

Each example is one iTHOR room rendered twice along the same camera path: before (S0) and
after (S1) several objects were moved, added or removed. The frames (RGB, depth, camera
pose) are streamed from the Hugging Face Hub by ``VlmCdDatasetMapper``; this module only
holds the light-weight parts:

- the **index** (``$VLMCD_DATA_DIR/index.json``), built once by
  ``vlm_cd/prepare_vlm_cd.py`` in the parent repo. It stores, per example, the changes,
  the camera intrinsics and where the example's frames sit in the ``walkthrough_frames``
  shards, so the mapper reads exactly two parquet row groups per sample;
- the **splits** (``$VLMCD_DATA_DIR/{train,val,test}.txt``), one example id per line;
- the **questions and answers** built from the changes that are visible in the frames a
  sample uses;
- the detectron2 **registration** of ``vlmcd_<task>_<split>`` datasets.

Tasks:

``describe``
    "List every object that was moved, added or removed." The answer is one clause per
    change, e.g. ``moved the vase on the diningtable by 6.4 m; removed the statue from the
    shelfmesh``.
``count``
    "How many objects were moved, added and removed?" e.g. ``7 moved, 1 added, 1 removed``.
"""

import json
import os
import re
from collections import Counter

from detectron2.data import DatasetCatalog, MetadataCatalog

REPO = "DR8803VL/vlm-change-detection-4k"
TASKS = ("describe", "count")
SPLITS = ("train", "val", "test")
ACTIONS = ("move", "add", "remove")
PAST = {"move": "moved", "add": "added", "remove": "removed"}

_INDEX_CACHE = {}


def data_dir():
    """Directory holding ``index.json`` and the split files.

    Returns
    -------
    str
        ``$VLMCD_DATA_DIR``, or ``data/vlm_cd`` next to the qwen3d checkout.
    """
    default = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "vlm_cd")
    return os.path.abspath(os.environ.get("VLMCD_DATA_DIR", default))


def load_index(root=None):
    """Load (and memoise) the example index.

    Parameters
    ----------
    root : str, optional
        Directory with ``index.json``. Defaults to ``data_dir()``.

    Returns
    -------
    dict
        ``{"repo", "revision", "examples": {example_id: record}}``.
    """
    root = root or data_dir()
    if root not in _INDEX_CACHE:
        path = os.path.join(root, "index.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; build it with `python vlm_cd/prepare_vlm_cd.py` "
                f"or set VLMCD_DATA_DIR")
        with open(path) as f:
            index = json.load(f)
        # The index keeps the raw dataset fields; names are cleaned here, in one place.
        for rec in index["examples"].values():
            for c in rec["changes"]:
                c["category"] = clean_category(c.get("raw_category"), c["object"])
                c["surface"] = clean_surface(c.get("support_surface"))
        _INDEX_CACHE[root] = index
    return _INDEX_CACHE[root]


def load_split(split, root=None):
    """Example ids of one split.

    Parameters
    ----------
    split : str
        ``"train"``, ``"val"`` or ``"test"``.
    root : str, optional
        Directory with the split files. Defaults to ``data_dir()``.

    Returns
    -------
    list of str
    """
    root = root or data_dir()
    with open(os.path.join(root, f"{split}.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def clean_surface(name):
    """Readable support-surface name.

    Parameters
    ----------
    name : str or None
        E.g. ``"diningtable_a1271102e7d3313087397a337003b3b7_1_0_0"`` or
        ``"standardwindow1(3)(2)_..."`` or ``"fp301:standardwindow_..."``.

    Returns
    -------
    str or None
        E.g. ``"diningtable"``, ``"standardwindow"``; ``None`` when there is no surface.
    """
    if not name:
        return None
    base = name.split("_")[0].split(":")[-1]  # "fp301:standardwindow..." -> "standardwindow"
    base = re.sub(r"\(.*?\)", "", base)
    base = re.sub(r"\d+$", "", base).strip()
    return base or None


def clean_category(category, obj=None):
    """Readable object category.

    Parameters
    ----------
    category : str or None
        The change's ``category``.
    obj : str, optional
        The change's ``object`` (``"vase_1"``), used when ``category`` is missing.

    Returns
    -------
    str
        Lower-case letters and digits only, so ``parse_describe`` can read it back.
    """
    name = category or re.sub(r"_\d+$", "", obj or "object")
    return re.sub(r"[^a-z0-9]", "", name.lower()) or "object"


def change_clause(change):
    """One clause of a ``describe`` answer.

    Parameters
    ----------
    change : dict
        An index change record: ``action``, ``category``, ``surface``, ``displacement_m``.

    Returns
    -------
    str
    """
    cat, surf = change["category"], change["surface"]
    if change["action"] == "move":
        text = f"moved the {cat}"
        if surf:
            text += f" on the {surf}"
        if change.get("displacement_m") is not None:
            text += f" by {change['displacement_m']:.1f} m"
        return text
    if change["action"] == "add":
        return f"added a {cat}" + (f" on the {surf}" if surf else "")
    return f"removed the {cat}" + (f" from the {surf}" if surf else "")


STATE_ENCODINGS = ("time", "offset")


def question(task, state_encoding="time"):
    """The prompt for a task.

    How the two scans reach the model depends on ``VLMCD.STATE_ENCODING`` (see
    ``VlmCdDatasetMapper``):

    ``time``
        two point clouds, each in its own ``<|vision_start|>...<|vision_end|>`` block, before
        first, both at the room's true coordinates and told apart by their time position.
    ``offset``
        one point cloud holding the room twice, the after scan shifted along +x.

    Parameters
    ----------
    task : str
        One of ``TASKS``.
    state_encoding : str
        One of ``STATE_ENCODINGS``.

    Returns
    -------
    str
    """
    if state_encoding == "time":
        intro = ("The two point clouds show the same room scanned twice: the first before "
                 "and the second after some objects changed. ")
    elif state_encoding == "offset":
        intro = ("The point cloud shows the same room scanned twice: first before and then "
                 "after some objects changed; the after scan is shifted along the x axis. ")
    else:
        raise ValueError(f"unknown VLMCD.STATE_ENCODING {state_encoding!r}; expected one of {STATE_ENCODINGS}")
    if task == "describe":
        return intro + ("List every object that was moved, added or removed, "
                        "separated by semicolons.")
    if task == "count":
        return intro + "How many objects were moved, added and removed?"
    raise ValueError(task)


ORDERS = ("size", "dataset")


def order_changes(changes, order="size"):
    """Put changes in the order the ``describe`` answer lists them.

    The language-modelling loss scores the answer token by token, so an arbitrary order
    of an unordered set adds loss the model cannot reduce. ``size`` makes the order a
    function of the scene: grouped by action (moved, added, removed, the same order as the
    ``count`` answer), then largest object first (object-frame bounding-box volume, so
    rotating an object does not change its rank), with the clause text breaking ties.

    Parameters
    ----------
    changes : list of dict
        Index change records.
    order : str
        ``"size"``, or ``"dataset"`` for the order of the dataset's ``changes`` column.

    Returns
    -------
    list of dict
    """
    if order == "dataset":
        return list(changes)
    if order != "size":
        raise ValueError(f"unknown VLMCD.ANSWER_ORDER {order!r}; expected one of {ORDERS}")
    if any("volume_m3" not in c for c in changes):
        raise KeyError("index.json has no object sizes; rebuild it with "
                       "`python vlm_cd/prepare_vlm_cd.py` or set VLMCD.ANSWER_ORDER dataset")
    # Identical objects (two chairs of one set) tie on volume; the clause text then decides.
    return sorted(changes, key=lambda c: (ACTIONS.index(c["action"]), -(c["volume_m3"] or 0.0),
                                          change_clause(c)))


def answer(task, changes, order="size"):
    """The target answer for a task.

    Parameters
    ----------
    task : str
        One of ``TASKS``.
    changes : list of dict
        Index change records, in dataset order.
    order : str
        Clause order of ``describe`` answers, see ``order_changes``.

    Returns
    -------
    str
    """
    if task == "describe":
        if not changes:
            return "no changes are visible"
        return "; ".join(change_clause(c) for c in order_changes(changes, order))
    if task == "count":
        n = Counter(c["action"] for c in changes)
        return ", ".join(f"{n[a]} {PAST[a]}" for a in ACTIONS)
    raise ValueError(task)


def parse_describe(text):
    """Parse a ``describe`` answer back into ``(action, category)`` pairs.

    Parameters
    ----------
    text : str
        Model output or ground truth.

    Returns
    -------
    collections.Counter
        Counts of ``(action, category)``.
    """
    out = Counter()
    for clause in text.lower().split(";"):
        m = re.match(r"\s*(moved|added|removed)\s+(?:the|a|an)?\s*([a-z][a-z0-9]*)", clause)
        if m:
            action = {"moved": "move", "added": "add", "removed": "remove"}[m.group(1)]
            out[(action, m.group(2))] += 1
    return out


def parse_count(text):
    """Parse a ``count`` answer.

    Parameters
    ----------
    text : str

    Returns
    -------
    dict
        ``{action: int or None}``.
    """
    out = {}
    for a in ACTIONS:
        m = re.search(rf"(\d+)\s+{PAST[a]}", text.lower())
        out[a] = int(m.group(1)) if m else None
    return out


def dataset_task(dataset_name):
    """Task of a registered dataset name ``vlmcd_<task>_<split>``.

    Parameters
    ----------
    dataset_name : str

    Returns
    -------
    str
    """
    return dataset_name.split("_")[1]


def load_vlmcd(task, split, root=None):
    """Dataset dicts of one registered dataset.

    Parameters
    ----------
    task : str
        One of ``TASKS``.
    split : str
        One of ``SPLITS``.
    root : str, optional
        Directory with the index and split files.

    Returns
    -------
    list of dict
        One ``{"example_id", "task", "split"}`` per example; the mapper looks the rest up
        in the index.
    """
    index = load_index(root)["examples"]
    ids = [i for i in load_split(split, root) if i in index]
    return [{"example_id": i, "task": task, "split": split, "image_id": i} for i in ids]


def register_vlmcd(root=None):
    """Register ``vlmcd_<task>_<split>`` for every task and split.

    The index is read lazily, when a dataset is first used, so registering is free even
    when the index has not been built.

    Parameters
    ----------
    root : str, optional
        Directory with the index and split files. Defaults to ``data_dir()`` at load time.
    """
    for task in TASKS:
        for split in SPLITS:
            name = f"vlmcd_{task}_{split}"
            if name in DatasetCatalog:
                continue
            DatasetCatalog.register(name, lambda t=task, s=split: load_vlmcd(t, s, root))
            MetadataCatalog.get(name).set(thing_classes=["object"], evaluator_type="vqa")
