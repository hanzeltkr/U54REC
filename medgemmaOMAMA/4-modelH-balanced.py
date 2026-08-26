#!/usr/bin/env python3
"""
LoRA fine-tuning for MedGemma on BALANCED dataset (50/50 NonCancer/Cancer)

This version trains on undersampled data to prevent the model from
learning to predict NonCancer by default due to class imbalance.
"""

import os, math
import torch
from datasets import load_from_disk
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

import numpy as np
from PIL import Image as PILImage

# ----------------- Paths (BALANCED DATASET) -----------------
DATA_DIR = "/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/hf_proc_messages_balanced"  # ← BALANCED!
OUT_DIR  = "/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/medgemma_runs/omama1024_balanced"  # ← NEW output
HF_HOME  = "/hpcstor6/scratch01/a/anya.tongprasith001/.hf_cache"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)

# keep all HF caches off home
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "transformers"))

print("="*60)
print("🔬 TRAINING WITH BALANCED DATASET (50/50)")
print("="*60)
print(f"Data: {DATA_DIR}")
print(f"Output: {OUT_DIR}")
print("="*60 + "\n")

# ----------------- Model/processor -----------------
MODEL_ID = "google/medgemma-4b-it"
USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"

major_cc = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
if major_cc < 8:
    raise ValueError("Need BF16-capable GPU (Ampere+).")

#attn_impl = "flash_attention_2" if os.environ.get("USE_FLASH_ATTENTION_2") == "1" else "eager"
attn_impl = "sdpa"   # was: eager (default) / flash_attention_2 (opt-in)

bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

model_kwargs = dict(
    attn_implementation=attn_impl,
    torch_dtype=torch.bfloat16,
)

if bnb_config:
    model_kwargs["quantization_config"] = bnb_config

print(f"Loading {MODEL_ID} (attn={attn_impl}, 4bit={USE_4BIT})")
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)
processor = AutoProcessor.from_pretrained(MODEL_ID)
if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
    processor.tokenizer.padding_side = "right"

# classes
CLASS_NAMES = ["NonCancer", "Cancer"]
model.config.id2label = {i: c for i, c in enumerate(CLASS_NAMES)}
model.config.label2id = {c: i for i, c in enumerate(CLASS_NAMES)}
model.config.num_labels = len(CLASS_NAMES)

# ----------------- Load BALANCED dataset -----------------
data = load_from_disk(DATA_DIR)
train_ds = data["train"]
eval_ds  = data["validation"].shuffle(seed=123).select(range(min(1000, len(data["validation"]))))

print("\n" + "="*60)
print("📊 DATASET DISTRIBUTION")
print("="*60)
from collections import Counter
train_dist = Counter(train_ds["label"])
eval_dist = Counter(eval_ds["label"])

print(f"Train set ({len(train_ds):,} samples):")
for label_idx, count in sorted(train_dist.items()):
    print(f"  {CLASS_NAMES[label_idx]}: {count:,} ({count/len(train_ds)*100:.1f}%)")

print(f"\nEval set ({len(eval_ds):,} samples):")
for label_idx, count in sorted(eval_dist.items()):
    print(f"  {CLASS_NAMES[label_idx]}: {count:,} ({count/len(eval_ds)*100:.1f}%)")
print("="*60 + "\n")

# ----------------- LoRA config -----------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)

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

# ----------------- Data collator -----------------
def collate_fn(examples: list[dict]):
    texts = []
    images = []
    for ex in examples:
        images.append([load_npz_as_pil(ex["image_path"])])
        txt = processor.apply_chat_template(
            ex["messages"],
            add_generation_prompt=False,
            tokenize=False
        ).strip()
        texts.append(txt)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Labels = input_ids, mask padding and image tokens
    labels = batch["input_ids"].clone()

    # mask pad
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    # mask image tokens
    special = processor.tokenizer.special_tokens_map
    img_token_ids = set()
    for k in ("boi_token", "eoi_token", "image_token"):
        tok = special.get(k, None)
        if tok is not None:
            tid = processor.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != processor.tokenizer.unk_token_id:
                img_token_ids.add(tid)
    img_token_ids.add(262144)

    for tid in img_token_ids:
        labels[labels == tid] = -100

    batch["labels"] = labels
    return batch

# ----------------- Training args -----------------
# With balanced data, we don't need class weights!
# Standard training should work well now.

args = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=24,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    learning_rate=2e-4,
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    lr_scheduler_type="linear",
    bf16=True,
    logging_steps=50,
    eval_steps=500,
    report_to="none",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    dataloader_num_workers=4,
    label_names=["labels"],
    push_to_hub=False,
    save_strategy="steps",
    save_steps=500,  # More frequent saves since dataset is smaller
    save_total_limit=3
)

# ----------------- Trainer -----------------
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# ----------------- Training -----------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING WITH BALANCED DATA")
    print("="*60)
    print("Expected improvements:")
    print("  ✅ No class imbalance bias")
    print("  ✅ Model learns to detect both classes equally")
    print("  ✅ Should see 75-85% sensitivity (vs 16% before)")
    print("  ✅ Slightly lower overall accuracy (90-92% vs 96%)")
    print("  ✅ But much better clinical utility!")
    print("="*60 + "\n")
    
    trainer.train()
    
    print("\n" + "="*60)
    print("💾 SAVING MODEL")
    print("="*60)
    trainer.save_model(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    print(f"✓ Done. Output in: {OUT_DIR}")
    print("\n💡 Next step:")
    print(f"  Update evaluation script to use: {OUT_DIR}")
    print("="*60 + "\n")
