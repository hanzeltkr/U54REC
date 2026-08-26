#!/usr/bin/env python3
"""
Final evaluation script with:
1. Lenient parser (DataCamp-style substring matching)
2. Debug logging to inspect raw model outputs
3. Label distribution verification
4. Detailed rejection tracking
"""
import os
import json
import re
from datetime import datetime
from typing import List, Tuple
from collections import Counter
import torch
from datasets import load_from_disk, ClassLabel
import evaluate
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
)

import numpy as np
from PIL import Image as PILImage

def load_npz_as_pil(path: str) -> PILImage:
    with np.load(path, mmap_mode="r") as data:
        arr = data["data"]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        a, b = np.percentile(arr, [0.5, 99.5])
        if b > a:
            arr = np.clip((arr - a) / (b - a), 0, 1)
        else:
            arr = np.clip(arr, 0, 1)
        arr = (arr * 255.0).round().astype(np.uint8)
    return PILImage.fromarray(arr, mode="RGB")

# ---------- paths ----------
DATA_DIR = "/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/omama/hf_proc_messages_balanced"
BASE_ID  = "google/medgemma-4b-it"   # baseline (gated—needs token/login)
FINETUNE_DIR = "/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/medgemma_runs/omama1024_balanced"  # your OUT_DIR
DEBUG_LOG_FILE = "/hpcstor6/scratch01/a/anya.tongprasith001/U54REC/medgemma_runs/evaluation_debug.jsonl"

# keep HF caches off home (optional but recommended)
os.environ.setdefault("HF_DATASETS_CACHE", "/hpcstor6/scratch01/a/anya.tongprasith001/.hf_cache/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/hpcstor6/scratch01/a/anya.tongprasith001/.hf_cache/transformers")

# ---------- load data ----------
data = load_from_disk(DATA_DIR)
test_data = data["validation"]  # already has 'image', 'label', 'messages'

# Detect classes (should be ['NonCancer', 'Cancer'])
CLASS_NAMES = test_data.features["label"].names
assert CLASS_NAMES == ["NonCancer", "Cancer"], f"Got {CLASS_NAMES}"
LABEL_FEATURE: ClassLabel = test_data.features["label"]

# Verify label distribution
print("\n" + "="*60)
print("📊 LABEL DISTRIBUTION VERIFICATION")
print("="*60)
label_counts = Counter(test_data["label"])
for label_idx, count in sorted(label_counts.items()):
    label_name = CLASS_NAMES[label_idx]
    pct = count / len(test_data) * 100
    print(f"  {label_name}: {count:,} samples ({pct:.1f}%)")
print(f"  Total: {len(test_data):,} samples")
print("="*60 + "\n")

# Build prompts (user turn only; model will complete the answer)
def chat_to_prompt(chat_turns):
    return processor.apply_chat_template(
        chat_turns,
        add_generation_prompt=True,  # tell model "assistant's turn"
        tokenize=False
    )

# ---------- metrics ----------
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_fscore_support
)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

REFERENCES = test_data["label"]

def compute_metrics(preds: List[int], verbose=True):
    """Compute comprehensive metrics for medical AI evaluation."""
    # Handle invalid predictions
    valid_preds = [p if p in (0,1) else -1 for p in preds]
    
    # Filter out invalid predictions for metrics
    valid_mask = np.array(valid_preds) != -1
    filtered_preds = np.array(valid_preds)[valid_mask]
    filtered_refs = np.array(REFERENCES)[valid_mask]
    
    # Basic metrics
    acc = accuracy_metric.compute(predictions=filtered_preds, references=filtered_refs)
    f1  = f1_metric.compute(predictions=filtered_preds, references=filtered_refs, average="weighted")
    
    # Confusion matrix
    cm = confusion_matrix(filtered_refs, filtered_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Per-class metrics
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        filtered_refs, filtered_preds, labels=[0, 1], zero_division=0
    )
    
    # Medical AI specific metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall for Cancer class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Same as recall for NonCancer
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0          # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0          # Negative Predictive Value
    
    # False rates
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Invalid prediction tracking
    n_invalid = len(preds) - len(filtered_preds)
    invalid_rate = n_invalid / len(preds) if len(preds) > 0 else 0
    
    metrics = {
        **acc, 
        **f1,
        "precision_noncancer": float(precision[0]),
        "recall_noncancer": float(recall[0]),
        "f1_noncancer": float(f1_per_class[0]),
        "support_noncancer": int(support[0]),
        
        "precision_cancer": float(precision[1]),
        "recall_cancer": float(recall[1]),
        "f1_cancer": float(f1_per_class[1]),
        "support_cancer": int(support[1]),
        
        "sensitivity": float(sensitivity),      # Cancer detection rate
        "specificity": float(specificity),      # NonCancer detection rate
        "ppv": float(ppv),                      # Precision for Cancer
        "npv": float(npv),                      # Precision for NonCancer
        
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        
        "n_invalid_predictions": int(n_invalid),
        "invalid_prediction_rate": float(invalid_rate),
    }
    
    if verbose:
        print("\n" + "="*60)
        print("DETAILED EVALUATION METRICS")
        print("="*60)
        print(f"\n📊 Overall Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.2%}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  Invalid predictions: {n_invalid}/{len(preds)} ({invalid_rate:.2%})")
        
        print(f"\n🎯 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"               NonCancer  Cancer")
        print(f"  Actual NonC    {tn:5d}    {fp:5d}")
        print(f"         Cancer  {fn:5d}    {tp:5d}")
        
        print(f"\n🔬 Per-Class Performance:")
        print(f"  NonCancer:")
        print(f"    Precision: {precision[0]:.2%}  (When model says NonCancer, correct {precision[0]:.1%} of time)")
        print(f"    Recall:    {recall[0]:.2%}  (Catches {recall[0]:.1%} of actual NonCancer cases)")
        print(f"    F1-Score:  {f1_per_class[0]:.4f}")
        print(f"    Support:   {support[0]} samples")
        
        print(f"\n  Cancer:")
        print(f"    Precision: {precision[1]:.2%}  (When model says Cancer, correct {precision[1]:.1%} of time)")
        print(f"    Recall:    {recall[1]:.2%}  (Catches {recall[1]:.1%} of actual Cancer cases) ⚠️ CRITICAL")
        print(f"    F1-Score:  {f1_per_class[1]:.4f}")
        print(f"    Support:   {support[1]} samples")
        
        print(f"\n🏥 Medical AI Metrics:")
        print(f"  Sensitivity (Cancer Recall): {sensitivity:.2%}  ← How many cancers detected")
        print(f"  Specificity (NonC Recall):   {specificity:.2%}  ← How many non-cancers correct")
        print(f"  PPV (Cancer Precision):      {ppv:.2%}  ← Positive result reliability")
        print(f"  NPV (NonCancer Precision):   {npv:.2%}  ← Negative result reliability")
        
        print(f"\n⚠️  Error Analysis:")
        print(f"  False Positive Rate: {false_positive_rate:.2%}  ({fp} false alarms)")
        print(f"  False Negative Rate: {false_negative_rate:.2%}  ({fn} missed cancers) ⚠️ CRITICAL")
        
        print(f"\n💡 Clinical Interpretation:")
        if false_negative_rate < 0.05:
            print(f"  ✅ Excellent: Missing <5% of cancer cases")
        elif false_negative_rate < 0.10:
            print(f"  ✓  Good: Missing <10% of cancer cases")
        elif false_negative_rate < 0.15:
            print(f"  ⚠️  Fair: Missing 10-15% of cancer cases")
        else:
            print(f"  ❌ Poor: Missing >15% of cancer cases - needs improvement")
        
        if false_positive_rate < 0.10:
            print(f"  ✅ Low false alarm rate (<10%)")
        elif false_positive_rate < 0.20:
            print(f"  ✓  Acceptable false alarm rate (<20%)")
        else:
            print(f"  ⚠️  High false alarm rate (>{false_positive_rate:.0%})")
        
        print("="*60 + "\n")
    
    return metrics

# ---------- LENIENT PARSER (DataCamp-style) WITH NEGATION HANDLING ----------
def postprocess_text_to_label_lenient(text: str, true_label: int = None, debug_log: list = None) -> Tuple[int, str]:
    """
    Lenient parser that accepts substring matches with normalization.
    CRITICAL: Handles negations like "no cancer", "not cancer", "no signs of cancer"
    Returns: (predicted_label, reason)
    
    Based on DataCamp tutorial approach with improvements for robustness.
    """
    raw_text = text  # Keep original for logging
    t = text.strip().lower()  # Normalize: lowercase and strip
    
    # Priority 1: Check for "B: Cancer" or "B:Cancer" or "B Cancer" patterns (exact format from training)
    if re.search(r'b\s*:?\s*cancer', t):
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text,
                "normalized": t,
                "predicted": 1,
                "true_label": true_label,
                "reason": "matched_pattern_b_cancer",
                "correct": true_label == 1 if true_label is not None else None
            })
        return 1, "matched_pattern_b_cancer"
    
    # Priority 2: Check for "A: NonCancer" or "A:NonCancer" or "A NonCancer" patterns
    if re.search(r'a\s*:?\s*noncancer', t):
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text,
                "normalized": t,
                "predicted": 0,
                "true_label": true_label,
                "reason": "matched_pattern_a_noncancer",
                "correct": true_label == 0 if true_label is not None else None
            })
        return 0, "matched_pattern_a_noncancer"
    
    # Priority 3: Check for NEGATIONS before checking for "cancer"
    # Patterns like "no cancer", "not cancer", "no obvious signs of cancer", "no signs of cancer"
    negation_patterns = [
        r'non-?cancerous',  # "non-cancerous", "noncancerous" - MUST BE FIRST!
        r'no\s+(?:obvious\s+)?(?:signs?\s+of\s+)?cancer',  # "no cancer", "no signs of cancer", "no obvious signs of cancer"
        r'not\s+cancer',  # "not cancer"
        r'no\s+(?:evidence|indication)\s+of\s+cancer',  # "no evidence of cancer"
        r'does\s+not\s+(?:appear|seem)\s+(?:to\s+(?:be|have)\s+)?cancer',  # "does not appear to be cancer"
        r'doesn?\'?t\s+suggest\s+(?:malignancy|cancer)',  # "doesn't suggest malignancy", "don't suggest malignancy"
        r'negative\s+for\s+cancer',  # "negative for cancer"
        r'normal.*no.*cancer',  # "normal... no cancer"
        r'absence\s+of.*(?:masses|cancer)',  # "absence of any obvious masses"
        r'normal\s+finding',  # "normal finding"
        r'benign',  # benign = not cancer
    ]
    
    for pattern in negation_patterns:
        if re.search(pattern, t):
            if debug_log is not None:
                debug_log.append({
                    "raw_output": raw_text,
                    "normalized": t,
                    "predicted": 0,
                    "true_label": true_label,
                    "reason": f"negation_detected: {pattern}",
                    "correct": true_label == 0 if true_label is not None else None
                })
            return 0, f"negation_detected: {pattern}"
    
    # Priority 4: Check for substring "noncancer" (must come before general "cancer")
    if "noncancer" in t:
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text,
                "normalized": t,
                "predicted": 0,
                "true_label": true_label,
                "reason": "substring_noncancer",
                "correct": true_label == 0 if true_label is not None else None
            })
        return 0, "substring_noncancer"
    
    # Priority 5: Simple substring match for "cancer" (but only if NOT negated above)
    # Check for positive cancer indicators
    positive_cancer_patterns = [
        r'(?:suspicious|possible|probable|likely)\s+(?:\w+\s+){0,3}cancer',  # "suspicious cancer", "likely cancer"
        r'(?:suggests?|indicates?)\s+(?:\w+\s+){0,3}(?:malignancy|cancer)',  # "suggests cancer/malignancy"
        r'classification\s+is\s+(?:\*\*)?b\s*:?\s*cancer',  # "classification is B: Cancer"
        r'therefore.*b\s*:?\s*cancer',  # "therefore B: Cancer"
    ]
    
    for pattern in positive_cancer_patterns:
        if re.search(pattern, t):
            if debug_log is not None:
                debug_log.append({
                    "raw_output": raw_text,
                    "normalized": t,
                    "predicted": 1,
                    "true_label": true_label,
                    "reason": f"positive_cancer_pattern: {pattern}",
                    "correct": true_label == 1 if true_label is not None else None
                })
            return 1, f"positive_cancer_pattern: {pattern}"
    
    # Priority 6: Generic "cancer" substring (fallback - use cautiously)
    # Only match if it's not already caught by negations above
    if "cancer" in t:
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text,
                "normalized": t,
                "predicted": 1,
                "true_label": true_label,
                "reason": "substring_cancer_generic",
                "correct": true_label == 1 if true_label is not None else None
            })
        return 1, "substring_cancer_generic"
    
    # Priority 7: Check for just "A" or "B" alone (sometimes model outputs just the letter)
    if re.search(r'\bb\b', t):  # \b for word boundary
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text,
                "normalized": t,
                "predicted": 1,
                "true_label": true_label,
                "reason": "matched_letter_b",
                "correct": true_label == 1 if true_label is not None else None
            })
        return 1, "matched_letter_b"
    
    if re.search(r'\ba\b', t):
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text,
                "normalized": t,
                "predicted": 0,
                "true_label": true_label,
                "reason": "matched_letter_a",
                "correct": true_label == 0 if true_label is not None else None
            })
        return 0, "matched_letter_a"
    
    # Failed to parse - mark as invalid
    if debug_log is not None:
        debug_log.append({
            "raw_output": raw_text,
            "normalized": t,
            "predicted": -1,
            "true_label": true_label,
            "reason": "unparseable",
            "correct": None
        })
    return -1, "unparseable"

# ---------- batch predict WITH DEBUG LOGGING ----------
def batch_predict(prompts, images, model, processor, *, batch_size=32, device="cuda", 
                  dtype=torch.bfloat16, debug_log=None, **gen_kwargs):
    """
    Batch prediction with optional debug logging.
    
    Args:
        debug_log: If provided, will collect debug info for all predictions
    """
    preds = []
    raw_outputs = []  # Store raw decoded text
    
    for i in range(0, len(prompts), batch_size):
        texts = prompts[i: i+batch_size]
        imgs  = [[im] for im in images[i: i+batch_size]]  # list of lists (one image)
        enc = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(device)
        
        # cast only floating tensors to dtype
        for k, v in enc.items():
            if torch.is_floating_point(v):
                enc[k] = v.to(dtype)

        # record prompt lengths to slice generation
        plen = enc["input_ids"].shape[1]
        
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=40,
                do_sample=False,
                **gen_kwargs
            )
        
        # decode continuation only
        for idx, seq in enumerate(out):
            ans = processor.decode(seq[plen:], skip_special_tokens=True)
            raw_outputs.append(ans)
            
            # Get true label for this sample
            true_label = REFERENCES[i + idx]
            
            # Use lenient parser with debug logging
            pred, reason = postprocess_text_to_label_lenient(ans, true_label, debug_log)
            preds.append(pred)
    
    return preds, raw_outputs

# ---------- DEBUG: Sample raw outputs ----------
def analyze_raw_outputs(raw_outputs, predictions, references, n_samples=10):
    """Analyze raw model outputs to understand prediction patterns."""
    print("\n" + "="*60)
    print("🔍 RAW OUTPUT ANALYSIS (Sample)")
    print("="*60)
    
    # Show examples of each category
    categories = {
        "Correct NonCancer": [],
        "Correct Cancer": [],
        "Incorrect NonCancer": [],
        "Incorrect Cancer": [],
        "Invalid": []
    }
    
    for i, (raw, pred, ref) in enumerate(zip(raw_outputs, predictions, references)):
        if pred == -1:
            categories["Invalid"].append((i, raw, pred, ref))
        elif pred == ref:
            if ref == 0:
                categories["Correct NonCancer"].append((i, raw, pred, ref))
            else:
                categories["Correct Cancer"].append((i, raw, pred, ref))
        else:
            if pred == 0:
                categories["Incorrect NonCancer"].append((i, raw, pred, ref))
            else:
                categories["Incorrect Cancer"].append((i, raw, pred, ref))
    
    # Print samples from each category
    for category, samples in categories.items():
        print(f"\n📌 {category} ({len(samples)} total):")
        for i, (idx, raw, pred, ref) in enumerate(samples[:n_samples]):
            true_label = CLASS_NAMES[ref]
            pred_label = CLASS_NAMES[pred] if pred in (0, 1) else "INVALID"
            print(f"  [{i+1}] Sample {idx}: True={true_label}, Pred={pred_label}")
            print(f"      Output: {repr(raw[:200])}")  # Show first 200 chars
    
    print("="*60 + "\n")

# ---------- load baseline ----------
print("\n" + "="*60)
print("🔵 LOADING BASELINE MODEL")
print("="*60)
processor = AutoProcessor.from_pretrained(BASE_ID)
tok = processor.tokenizer
# Fix padding side for decoder-only generation
tok.padding_side = "left"

base = AutoModelForImageTextToText.from_pretrained(
    BASE_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda").eval()

# Set generation config
gen_cfg = GenerationConfig.from_pretrained(BASE_ID)
gen_cfg.update(
    do_sample=False,
    top_k=None,
    top_p=None,
    cache_implementation="dynamic",
)
base.generation_config = gen_cfg

# align pad ids with tokenizer
base.config.pad_token_id            = tok.pad_token_id
base.generation_config.pad_token_id = tok.pad_token_id

# Build prompts/images
prompts = [chat_to_prompt(c) for c in test_data["messages"]]
images = [load_npz_as_pil(p) for p in test_data["image_path"]]

print(f"Eval set: {len(prompts)} samples")
print(f"Using lenient parser (DataCamp-style)")
print("="*60 + "\n")

# ---------- baseline predictions WITH DEBUG ----------
print("Running baseline inference with debug logging...")
baseline_debug_log = []
bf_preds, bf_raw_outputs = batch_predict(
    prompts, images, base, processor, 
    batch_size=8, 
    debug_log=baseline_debug_log
)

print("\n" + "🔵 BASELINE MODEL EVALUATION " + "🔵")
bf_metrics = compute_metrics(bf_preds, verbose=True)

# Analyze raw outputs
analyze_raw_outputs(bf_raw_outputs, bf_preds, REFERENCES, n_samples=5)

# ---------- load fine-tuned ----------

del base
import gc; gc.collect()
torch.cuda.empty_cache()

print("\n" + "="*60)
print("🟢 LOADING FINE-TUNED MODEL")
print("="*60)
processor_ft = AutoProcessor.from_pretrained(FINETUNE_DIR)
tok_ft = processor_ft.tokenizer
# Fix padding side for decoder-only generation
tok_ft.padding_side = "left"

ft = AutoModelForImageTextToText.from_pretrained(
    FINETUNE_DIR,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda").eval()

# Use same style generation
ft.generation_config = gen_cfg
ft.config.pad_token_id            = tok_ft.pad_token_id
ft.generation_config.pad_token_id = tok_ft.pad_token_id
print("="*60 + "\n")

# ---------- fine-tuned predictions WITH DEBUG ----------
print("Running fine-tuned inference with debug logging...")
finetuned_debug_log = []
af_preds, af_raw_outputs = batch_predict(
    prompts, images, ft, processor_ft, 
    batch_size=8,
    debug_log=finetuned_debug_log
)

print("\n" + "🟢 FINE-TUNED MODEL EVALUATION " + "🟢")
af_metrics = compute_metrics(af_preds, verbose=True)

# Analyze raw outputs
analyze_raw_outputs(af_raw_outputs, af_preds, REFERENCES, n_samples=5)

# ---------- SAVE DEBUG LOGS ----------
print("\n" + "="*60)
print("💾 SAVING DEBUG LOGS")
print("="*60)

# Make sure if have directory
os.makedirs(os.path.dirname(DEBUG_LOG_FILE), exist_ok=True)

# Save baseline debug log
baseline_debug_file = DEBUG_LOG_FILE.replace(".jsonl", "_baseline.jsonl")
with open(baseline_debug_file, 'w') as f:
    for entry in baseline_debug_log:
        f.write(json.dumps(entry) + '\n')
print(f"Baseline debug log saved to: {baseline_debug_file}")
print(f"  Total entries: {len(baseline_debug_log)}")

# Save finetuned debug log
finetuned_debug_file = DEBUG_LOG_FILE.replace(".jsonl", "_finetuned.jsonl")
with open(finetuned_debug_file, 'w') as f:
    for entry in finetuned_debug_log:
        f.write(json.dumps(entry) + '\n')
print(f"Fine-tuned debug log saved to: {finetuned_debug_file}")
print(f"  Total entries: {len(finetuned_debug_log)}")

# Analyze debug logs
print("\n📊 Debug Log Summary:")
print("\nBaseline parser reasons:")
baseline_reasons = Counter(entry['reason'] for entry in baseline_debug_log)
for reason, count in baseline_reasons.most_common():
    print(f"  {reason}: {count} ({count/len(baseline_debug_log)*100:.1f}%)")

print("\nFine-tuned parser reasons:")
finetuned_reasons = Counter(entry['reason'] for entry in finetuned_debug_log)
for reason, count in finetuned_reasons.most_common():
    print(f"  {reason}: {count} ({count/len(finetuned_debug_log)*100:.1f}%)")

print("="*60 + "\n")

# ---------- comparison summary ----------
print("\n" + "="*60)
print("📈 IMPROVEMENT SUMMARY (Baseline → Fine-tuned)")
print("="*60)

def format_delta(base_val, ft_val, percentage=True, inverse=False):
    """Format improvement delta with color coding."""
    delta = ft_val - base_val
    if inverse:  # For metrics where lower is better (false rates)
        delta = -delta
    
    symbol = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
    sign = "+" if delta > 0 else ""
    
    if percentage:
        return f"{symbol} {sign}{delta:+.2%} ({base_val:.2%} → {ft_val:.2%})"
    else:
        return f"{symbol} {sign}{delta:+.4f} ({base_val:.4f} → {ft_val:.4f})"

print(f"\nOverall:")
print(f"  Accuracy:     {format_delta(bf_metrics['accuracy'], af_metrics['accuracy'])}")
print(f"  F1-Score:     {format_delta(bf_metrics['f1'], af_metrics['f1'], percentage=False)}")

print(f"\nCancer Detection (Most Critical):")
print(f"  Sensitivity:  {format_delta(bf_metrics['sensitivity'], af_metrics['sensitivity'])}")
print(f"  Precision:    {format_delta(bf_metrics['precision_cancer'], af_metrics['precision_cancer'])}")
print(f"  F1-Score:     {format_delta(bf_metrics['f1_cancer'], af_metrics['f1_cancer'], percentage=False)}")

print(f"\nError Rates (Lower is Better):")
print(f"  False Negatives: {format_delta(bf_metrics['false_negative_rate'], af_metrics['false_negative_rate'], inverse=True)}")
print(f"  False Positives: {format_delta(bf_metrics['false_positive_rate'], af_metrics['false_positive_rate'], inverse=True)}")

print(f"\nPrediction Quality:")
print(f"  Invalid preds (baseline):    {bf_metrics['n_invalid_predictions']}/{len(bf_preds)} ({bf_metrics['invalid_prediction_rate']:.2%})")
print(f"  Invalid preds (fine-tuned):  {af_metrics['n_invalid_predictions']}/{len(af_preds)} ({af_metrics['invalid_prediction_rate']:.2%})")

# Overall assessment
accuracy_gain = af_metrics['accuracy'] - bf_metrics['accuracy']
sensitivity_gain = af_metrics['sensitivity'] - bf_metrics['sensitivity']

print(f"\n💡 Overall Assessment:")
if accuracy_gain > 0.10 and sensitivity_gain > 0.10:
    print("  🌟 EXCELLENT improvement! Fine-tuning significantly helped.")
elif accuracy_gain > 0.05 or sensitivity_gain > 0.05:
    print("  ✅ GOOD improvement. Fine-tuning was beneficial.")
elif accuracy_gain > 0 or sensitivity_gain > 0:
    print("  ✓  MODEST improvement. Fine-tuning helped slightly.")
elif accuracy_gain > -0.02 and sensitivity_gain > -0.02:
    print("  ➡️  MINIMAL change. Baseline was already strong.")
else:
    print("  ⚠️  DEGRADATION. Fine-tuning may have hurt performance.")

print("="*60 + "\n")

# ---------- save results ----------
results_file = os.path.join(
    os.path.dirname(FINETUNE_DIR), 
    f"final_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)

results = {
    "timestamp": datetime.now().isoformat(),
    "data_dir": DATA_DIR,
    "base_model": BASE_ID,
    "finetune_dir": FINETUNE_DIR,
    "n_samples": len(prompts),
    "class_names": CLASS_NAMES,
    "label_distribution": {CLASS_NAMES[k]: int(v) for k, v in label_counts.items()},
    "parser_type": "lenient_datacamp_style",
    "baseline_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                         for k, v in bf_metrics.items()},
    "finetuned_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                          for k, v in af_metrics.items()},
    "improvements": {
        "accuracy_delta": float(af_metrics['accuracy'] - bf_metrics['accuracy']),
        "sensitivity_delta": float(af_metrics['sensitivity'] - bf_metrics['sensitivity']),
        "specificity_delta": float(af_metrics['specificity'] - bf_metrics['specificity']),
        "f1_delta": float(af_metrics['f1'] - bf_metrics['f1']),
        "false_negative_rate_delta": float(bf_metrics['false_negative_rate'] - af_metrics['false_negative_rate']),
    },
    "parser_statistics": {
        "baseline": dict(baseline_reasons),
        "finetuned": dict(finetuned_reasons)
    },
    "debug_files": {
        "baseline": baseline_debug_file,
        "finetuned": finetuned_debug_file
    }
}

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"📁 Results saved to: {results_file}")
print(f"\n✅ Final evaluation complete!")
print(f"\n📝 Next steps:")
print(f"  1. Review confusion matrix - should see Cancer predictions now")
print(f"  2. Check sensitivity - target >85% for medical AI")
print(f"  3. Analyze debug logs to understand model behavior:")
print(f"     - {baseline_debug_file}")
print(f"     - {finetuned_debug_file}")
print(f"  4. If sensitivity still low, investigate training data quality\n")
