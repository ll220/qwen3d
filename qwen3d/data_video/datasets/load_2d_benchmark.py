from datasets import load_dataset, concatenate_datasets
import pandas as pd
from datasets.arrow_dataset import Dataset as HFDataset
import os
import json
import re


# List of all BLINK subtasks
BLINK_SUBTASKS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]

MMU_SUBTASKS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art",
    "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry",
    "Clinical_Medicine", "Computer_Science", "Design", "Diagnostics_and_Laboratory_Medicine",
    "Economics", "Electronics", "Energy_and_Power", "Finance", "Geography",
    "History", "Literature", "Manage", "Marketing", "Materials",
    "Math", "Mechanical_Engineering", "Music", "Pharmacy", "Physics",
    "Psychology", "Public_Health", "Sociology"
]

def load_blink_val():
    val_sets = []
    for task in BLINK_SUBTASKS:
        ds = load_dataset("BLINK-Benchmark/BLINK", task, split="val")
        val_sets.append(ds)
    return concatenate_datasets(val_sets)

def load_realworld_vqa_val():
    dataset = load_dataset("xai-org/RealworldQA", split='test')
    return dataset

def load_alpaca():
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    return dataset

def load_mmmu_val():
    val_sets = []
    for subject in MMU_SUBTASKS:
        ds = load_dataset("MMMU/MMMU", subject, split="validation")
        val_sets.append(ds)
    return concatenate_datasets(val_sets)

def load_mmmlupro_val():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    return ds


def load_llava150k():
    """
    Load and expand LLaVA-Instruct-150K JSON(s) into single-turn samples.
    
    Args:
        data_dir (str or Path): directory containing the JSONs
        use_full (bool): if True, load llava_instruct_150k.json (merged);
                         if False, load 3 main subsets: 77k, 58k, 23k.
    
    Returns:
        ds (datasets.Dataset): Hugging Face Dataset with expanded single-turn entries
    """
    def _strip_image_tokens(s: str) -> str:
        # Remove any <image>, <image 1>, <image_2> variants (case-insensitive)
        return re.sub(r"\s*<image[^>]*>\s*", " ", s, flags=re.IGNORECASE).strip()

    coco_path = os.path.join(os.environ.get("COCO_REF_DATASET"),  "coco", "train2017")
    llava_instruct_path = os.environ.get("LLAVA_INSTRUCT_PATH")
    with open(llava_instruct_path, "r") as f:
        data = json.load(f)

    all_examples = []
    for ex in data:
        image = ex.get("image")

        # --- Case 1: conversation-based sample ---
        if isinstance(ex.get("conversations"), list):
            convs = ex["conversations"]
            for i in range(0, len(convs) - 1, 2):
                user = convs[i]
                assistant = convs[i + 1]
                if (
                    user.get("from") == "human"
                    and assistant.get("from") == "gpt"
                ):
                    input_text = _strip_image_tokens(user["value"])
                    all_examples.append({
                        "image": image,
                        "question": input_text,
                        "answer": assistant["value"],
                        "image_file": os.path.join(coco_path, image)
                    })

        # --- Case 2: simple instruction–response ---
        else:
            instr = ex["conversations"][0].get("value")
            resp = ex["conversations"][1].get("value")
            if instr and resp:
                instr = _strip_image_tokens(instr)
                all_examples.append({
                    "image": image,
                    "question": instr,
                    "answer": resp,
                    "image_file": os.path.join(coco_path, image)               
                    })

    # --- Convert to DataFrame and Dataset ---
    df = pd.DataFrame(all_examples)
    print(f"✅ Expanded into {len(df):,} single-turn samples from {len(data)} llava-instruct-150k samples")

    ds = HFDataset.from_pandas(df, preserve_index=False)
    return ds

LOAD_BENCHMARKS = {
    # "blink_bench": load_blink_val,
    "realworld_vqa_bench": load_realworld_vqa_val,
    "mmmu_bench": load_mmmu_val,
    "llava150k_bench": load_llava150k,
    "mmlupro_text_bench": load_mmmlupro_val,
    "alpaca_text_bench": load_alpaca,
}

