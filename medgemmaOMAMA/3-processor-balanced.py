#!/usr/bin/env python3
"""
Processor for BALANCED dataset - formats data with messages for MedGemma training
"""
from datasets import load_from_disk, DatasetDict
import string, os

# ---------- Paths (UPDATED FOR BALANCED DATASET) ----------
ARROW_DIR = "/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/hf_arrow_balanced"  # ← BALANCED!
PROC_DIR  = "/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/hf_proc_messages_balanced"  # ← NEW output

# keep HF caches in scratch
os.environ.setdefault("HF_DATASETS_CACHE", "/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/.hf_cache/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/.hf_cache/transformers")

# ---------- Load balanced dataset ----------
print("="*60)
print("Loading BALANCED dataset")
print("="*60)
data = load_from_disk(ARROW_DIR)
print("Loaded:", data)

# Verify balance
from collections import Counter
train_dist = Counter(data["train"]["label"])
val_dist = Counter(data["validation"]["label"])

CLASSES = data["train"].features["label"].names
print(f"\nDetected classes: {CLASSES}")
print(f"\nTrain distribution:")
for label_idx, count in sorted(train_dist.items()):
    print(f"  {CLASSES[label_idx]}: {count:,} ({count/len(data['train'])*100:.1f}%)")
print(f"\nValidation distribution:")
for label_idx, count in sorted(val_dist.items()):
    print(f"  {CLASSES[label_idx]}: {count:,} ({count/len(data['validation'])*100:.1f}%)")

# ---------- Format with A/B prefixes ----------
letters = list(string.ascii_uppercase)
CLASS_PROMPTS = [f"{letters[i]}: {name}" for i, name in enumerate(CLASSES)]
options_text = "\n".join(CLASS_PROMPTS)

# Prompt for mammogram classification
PROMPT = f"Classify this mammogram.\n{options_text}"

print(f"\nPrompt format:")
print(f"{'-'*60}")
print(PROMPT)
print(f"{'-'*60}\n")

# ---------- Format each example into chat-style messages ----------
def format_data(example):
    example["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": CLASS_PROMPTS[example["label"]]},
            ],
        },
    ]
    return example

formatted = DatasetDict({
    "train": data["train"].map(format_data, desc="Formatting train"),
    "validation": data["validation"].map(format_data, desc="Formatting val"),
})

# Quick peek
print("Sample message format:")
print(formatted["train"][0]["messages"])
print()

# ---------- Save formatted dataset ----------
os.makedirs(PROC_DIR, exist_ok=True)
formatted.save_to_disk(PROC_DIR)

print("="*60)
print(f"✅ Saved formatted BALANCED dataset to:")
print(f"   {PROC_DIR}")
print("="*60)
print(f"\n📊 Summary:")
print(f"  Train: {len(formatted['train']):,} samples (balanced 50/50)")
print(f"  Val:   {len(formatted['validation']):,} samples (balanced 50/50)")
print(f"\n💡 Next step:")
print(f"  Update 4-modelH.py to use:")
print(f"  DATA_DIR = '{PROC_DIR}'")
print("="*60 + "\n")
