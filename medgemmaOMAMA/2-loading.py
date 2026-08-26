# step6_model.py — load MedGemma 4B Instruct and its processor
import os
import torch
from huggingface_hub import login
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

# ---------- Auth (use ONE of these methods) ----------
# 1) If you've already run: huggingface-cli login  → this is enough.
# 2) Or export an env var in your shell / Slurm script:
#    export HUGGINGFACE_HUB_TOKEN=hf_xxx
HF_TOKEN = "" #os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)

# ---------- Model selection ----------
MODEL_ID = "google/medgemma-4b-it"  # tutorial checkpoint (gated)

# ---------- Device / dtype ----------
# Require BF16-capable GPU (Ampere+). 4090/A100/H100 are OK.
major_cc = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
if major_cc < 8:
    raise ValueError("This GPU does not support bfloat16. Use Ampere+ (e.g., A100/3090/4090/H100).")

# Attention implementation: try flash-attn if installed, else safe eager
attn_impl = "flash_attention_2" if os.environ.get("USE_FLASH_ATTENTION_2") == "1" else "eager"

# ---------- Optional: 4-bit quant (QLoRA-ready) ----------
USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"
bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

# ---------- Load model ----------
model_kwargs = dict(
    attn_implementation=attn_impl,
    torch_dtype=torch.bfloat16,
    device_map="auto",        # will place layers on GPU
)
if bnb_config:
    model_kwargs["quantization_config"] = bnb_config

print(f"Loading model: {MODEL_ID} (attn={attn_impl}, 4bit={USE_4BIT})")
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)

# ---------- Load processor ----------
processor = AutoProcessor.from_pretrained(MODEL_ID)
# Right padding avoids some batching issues
if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
    processor.tokenizer.padding_side = "right"

# ---------- Your labels (must match step 4 order) ----------
CLASS_NAMES = ["NonCancer", "Cancer"]
model.config.id2label = {i: c for i, c in enumerate(CLASS_NAMES)}
model.config.label2id = {c: i for i, c in enumerate(CLASS_NAMES)}
model.config.num_labels = len(CLASS_NAMES)

# ---------- Sanity print ----------
print("✓ Model & processor loaded")
print("  dtype:", model.dtype)
print("  classes:", model.config.id2label)

# ---------- (Optional) quick dry-run on one sample of your formatted dataset ----------
if os.environ.get("DRYRUN", "0") == "1":
  from datasets import load_from_disk

  DATA_DIR = "/hpcstor6/scratch01/e/edward.gaibor001/omamadata256/hf_proc_messages"
  ds = load_from_disk(DATA_DIR)
  ex = ds["train"][0]

  # Build text that includes the <image> token(s)
  chat_text = processor.apply_chat_template(
      ex["messages"],
      add_generation_prompt=False,
      tokenize=False,
  )

  enc = processor(
      images=ex["image"],
      text=chat_text,       # now the text includes the image token
      return_tensors="pt",
      padding=True,
  )

  enc = {k: v.to(model.device) for k, v in enc.items()}
  with torch.inference_mode():
      _ = model(**enc)
  print("✓ Dry-run forward pass ok")
