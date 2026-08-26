#!/usr/bin/env python3
"""
Dataset preparation with BALANCED CLASS DISTRIBUTION (Option 4: Undersample NonCancer)

Strategy: Undersample NonCancer to match Cancer count → 50/50 balanced dataset
This prevents the model from learning to predict NonCancer by default.
"""

import os, glob, json, random
from typing import List, Dict
import numpy as np
from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Features, Image, ClassLabel, Value

# ---------- Paths ----------
ROOT = "/raid/mpsych/OMAMA/DATA/data/2d_resized_256"
IMAGES_DIR = os.path.join(ROOT, "images")
META_DIR   = os.path.join(ROOT, "metadata")
SAVE_DIR   = "/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/hf_arrow_balanced"  # NEW: balanced dataset

# Keep HF caches in scratch
os.environ.setdefault("HF_DATASETS_CACHE", "/hpcstor6/scratch01/a/anya.tongprasith001/.hf_cache/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/hpcstor6/scratch01/a/anya.tongprasith001/.hf_cache/transformers")

# ---------- Label normalization ----------
LABEL_NORMALIZE = {
    "NonCancer": "NonCancer",
    "IndexCancer": "Cancer",
    "PreIndexCancer": "Cancer",
    "Unknown": None,  # drop
}
CLASS_NAMES = ["NonCancer", "Cancer"]
SPLIT_SEED = 42
TRAIN_PROP = 0.75  # 75/25 split

# ---------- NPZ -> PIL helper ----------
def load_npz_as_pil(path: str) -> PILImage:
    with np.load(path, mmap_mode="r") as data:
        arr = data["data"]
    # Ensure HxWxC (RGB)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    # Convert to uint8 [0..255]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        a, b = np.percentile(arr, [0.5, 99.5])
        if b > a:
            arr = np.clip((arr - a) / (b - a), 0, 1)
        else:
            arr = np.clip(arr, 0, 1)
        arr = (arr * 255.0).round().astype(np.uint8)
    return PILImage.fromarray(arr, mode="RGB")

# ---------- Scan files & pair with labels ----------
print("="*60)
print("STEP 1: Loading and normalizing labels")
print("="*60)

npz_files: List[str] = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.npz")))
assert npz_files, f"No .npz files found in {IMAGES_DIR}"

records: List[Dict] = []
missing_meta = 0
dropped_unknown = 0
kept_counts = {"NonCancer": 0, "Cancer": 0}

for p in npz_files:
    base = os.path.splitext(os.path.basename(p))[0]
    meta_path = os.path.join(META_DIR, base + ".json")
    if not os.path.exists(meta_path):
        missing_meta += 1
        continue
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        raw = meta.get("label", None)
        norm = LABEL_NORMALIZE.get(raw, None)
        if norm is None:
            dropped_unknown += 1
            continue
        records.append({"filename": base + ".npz", "label_name": norm, "image_path": p})
        kept_counts[norm] += 1
    except Exception:
        dropped_unknown += 1

print(f"Found {len(npz_files)} npz files")
print(f"Built {len(records)} labeled records")
print(f"  Missing metadata: {missing_meta}")
print(f"  Dropped unknown/invalid: {dropped_unknown}")
print(f"\nOriginal class distribution:")
print(f"  NonCancer: {kept_counts['NonCancer']:,} ({kept_counts['NonCancer']/len(records)*100:.1f}%)")
print(f"  Cancer:    {kept_counts['Cancer']:,} ({kept_counts['Cancer']/len(records)*100:.1f}%)")
print(f"  Imbalance ratio: {kept_counts['NonCancer']/kept_counts['Cancer']:.1f}:1")

# ---------- BALANCE CLASSES (Undersample NonCancer) ----------
print("\n" + "="*60)
print("STEP 2: Balancing classes (undersampling NonCancer)")
print("="*60)

random.seed(SPLIT_SEED)
by_class = {c: [] for c in CLASS_NAMES}
for r in records:
    by_class[r["label_name"]].append(r)

# Shuffle before sampling
for c in CLASS_NAMES:
    random.shuffle(by_class[c])

# Undersample NonCancer to match Cancer count
n_cancer = len(by_class["Cancer"])
n_noncancer_original = len(by_class["NonCancer"])

# Sample NonCancer to match Cancer (50/50 balance)
by_class["NonCancer"] = by_class["NonCancer"][:n_cancer]

print(f"Target: 50/50 balance")
print(f"  Cancer samples:    {n_cancer:,}")
print(f"  NonCancer (before): {n_noncancer_original:,}")
print(f"  NonCancer (after):  {len(by_class['NonCancer']):,}")
print(f"  Removed: {n_noncancer_original - len(by_class['NonCancer']):,} NonCancer samples")

# Combine balanced records
balanced_records = []
for c in CLASS_NAMES:
    balanced_records.extend(by_class[c])

random.shuffle(balanced_records)

balanced_counts = {"NonCancer": 0, "Cancer": 0}
for r in balanced_records:
    balanced_counts[r["label_name"]] += 1

print(f"\nBalanced dataset:")
print(f"  Total: {len(balanced_records):,} samples")
print(f"  NonCancer: {balanced_counts['NonCancer']:,} ({balanced_counts['NonCancer']/len(balanced_records)*100:.1f}%)")
print(f"  Cancer:    {balanced_counts['Cancer']:,} ({balanced_counts['Cancer']/len(balanced_records)*100:.1f}%)")
print(f"  Ratio: {balanced_counts['NonCancer']/balanced_counts['Cancer']:.2f}:1 ✅")

# ---------- Stratified 80/20 split ----------
print("\n" + "="*60)
print("STEP 3: Creating stratified train/val split")
print("="*60)

# Re-organize by class for stratified split
by_class = {c: [] for c in CLASS_NAMES}
for r in balanced_records:
    by_class[r["label_name"]].append(r)

train_records, val_records = [], []
for c in CLASS_NAMES:
    n = len(by_class[c])
    n_train = int(n * TRAIN_PROP)
    train_records += by_class[c][:n_train]
    val_records   += by_class[c][n_train:]

random.shuffle(train_records)
random.shuffle(val_records)

print(f"Train split: {len(train_records):,} samples")
train_counts = {"NonCancer": 0, "Cancer": 0}
for r in train_records:
    train_counts[r["label_name"]] += 1
print(f"  NonCancer: {train_counts['NonCancer']:,} ({train_counts['NonCancer']/len(train_records)*100:.1f}%)")
print(f"  Cancer:    {train_counts['Cancer']:,} ({train_counts['Cancer']/len(train_records)*100:.1f}%)")

print(f"\nValidation split: {len(val_records):,} samples")
val_counts = {"NonCancer": 0, "Cancer": 0}
for r in val_records:
    val_counts[r["label_name"]] += 1
print(f"  NonCancer: {val_counts['NonCancer']:,} ({val_counts['NonCancer']/len(val_records)*100:.1f}%)")
print(f"  Cancer:    {val_counts['Cancer']:,} ({val_counts['Cancer']/len(val_records)*100:.1f}%)")

# ---------- Build HuggingFace Dataset ----------
print("\n" + "="*60)
print("STEP 4: Building HuggingFace Dataset")
print("="*60)

features = Features({
    "image_path": Value("string"),
    "label": ClassLabel(names=CLASS_NAMES),
    "filename": Value("string"),
})

def make_dataset(recs: List[Dict]) -> Dataset:
    labels = [CLASS_NAMES.index(r["label_name"]) for r in recs]
    filenames = [r["filename"] for r in recs]
    paths = [r["image_path"] for r in recs]
    ds = Dataset.from_dict(
        {"filename": filenames, "label": labels, "image_path": paths},
        features=features,
    )
    return ds   # no .map(_load), no .cast() to Image() — nothing decoded here

dataset = DatasetDict({
    "train": make_dataset(train_records),
    "validation": make_dataset(val_records),
})

print("\n" + dataset.__str__())
print(f"\nSample: {dataset['train'][0]['filename']}")
print(f"  Label: {CLASS_NAMES[dataset['train'][0]['label']]}")
print(f"  Image path: {dataset['train'][0]['image_path']}")
_sanity_img = load_npz_as_pil(dataset['train'][0]['image_path'])
print(f"  Sanity-check decode OK — image size: {_sanity_img.size}")

# ---------- Save balanced dataset ----------
print("\n" + "="*60)
print("STEP 5: Saving balanced dataset")
print("="*60)

os.makedirs(SAVE_DIR, exist_ok=True)
dataset.save_to_disk(SAVE_DIR)

print(f"✅ Saved balanced dataset to: {SAVE_DIR}")
print(f"\n📊 Summary:")
print(f"  Original: {kept_counts['NonCancer']:,} NonCancer, {kept_counts['Cancer']:,} Cancer (imbalanced {kept_counts['NonCancer']/kept_counts['Cancer']:.1f}:1)")
print(f"  Balanced: {balanced_counts['NonCancer']:,} NonCancer, {balanced_counts['Cancer']:,} Cancer (balanced 1:1)")
print(f"  Train:    {len(train_records):,} samples (50/50 split)")
print(f"  Val:      {len(val_records):,} samples (50/50 split)")
print(f"\n💡 Next step:")
print(f"  Run: python 3-processor.py (update ARROW_DIR to use balanced dataset)")
print(f"\nReload later with:")
print(f'  from datasets import load_from_disk')
print(f'  dataset = load_from_disk("{SAVE_DIR}")')
print("="*60 + "\n")
