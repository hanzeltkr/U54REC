#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from typing import Tuple
import pickle, random, os, json
from pathlib import Path
from collections import Counter
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix


# In[ ]:
# Limit Tensorflow to starts out allocating very little memory, and as the program gets run and more GPU memory is needed
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# ---------- LENIENT PARSER WITH NEGATION HANDLING ----------
def postprocess_text_to_label_lenient(text: str, true_label: int = None, debug_log: list = None) -> Tuple[int, str]:
    """Lenient parser with negation handling."""
    raw_text = text
    t = text.strip().lower()

    # Priority 1: Exact format patterns
    if re.search(r'b\s*:?\s*cancer', t):
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 1, "true_label": true_label,
                "reason": "matched_pattern_b_cancer", "correct": true_label == 1 if true_label is not None else None
            })
        return 1, "matched_pattern_b_cancer"

    if re.search(r'a\s*:?\s*noncancer', t):
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 0, "true_label": true_label,
                "reason": "matched_pattern_a_noncancer", "correct": true_label == 0 if true_label is not None else None
            })
        return 0, "matched_pattern_a_noncancer"

    # Priority 2: Negation patterns (BEFORE generic cancer matching)
    negation_patterns = [
        r'non-?cancerous',
        r'no\s+(?:obvious\s+)?(?:signs?\s+of\s+)?cancer',
        r'not\s+cancer',
        r'no\s+(?:evidence|indication)\s+of\s+cancer',
        r'does\s+not\s+(?:appear|seem)\s+(?:to\s+(?:be|have)\s+)?cancer',
        r'doesn?\'?t\s+suggest\s+(?:malignancy|cancer)',
        r'negative\s+for\s+cancer',
        r'normal.*no.*cancer',
        r'absence\s+of.*(?:masses|cancer)',
        r'normal\s+finding',
        r'benign',
    ]

    for pattern in negation_patterns:
        if re.search(pattern, t):
            if debug_log is not None:
                debug_log.append({
                    "raw_output": raw_text, "normalized": t, "predicted": 0, "true_label": true_label,
                    "reason": f"negation: {pattern}", "correct": true_label == 0 if true_label is not None else None
                })
            return 0, f"negation: {pattern}"

    # Priority 3: NonCancer substring
    if "noncancer" in t:
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 0, "true_label": true_label,
                "reason": "substring_noncancer", "correct": true_label == 0 if true_label is not None else None
            })
        return 0, "substring_noncancer"

    # Priority 4: Positive cancer indicators
    positive_patterns = [
        r'(?:suspicious|possible|probable|likely)\s+(?:\w+\s+){0,3}cancer',
        r'(?:suggests?|indicates?)\s+(?:\w+\s+){0,3}(?:malignancy|cancer)',
        r'classification\s+is\s+(?:\*\*)?b\s*:?\s*cancer',
        r'therefore.*b\s*:?\s*cancer',
    ]

    for pattern in positive_patterns:
        if re.search(pattern, t):
            if debug_log is not None:
                debug_log.append({
                    "raw_output": raw_text, "normalized": t, "predicted": 1, "true_label": true_label,
                    "reason": f"positive: {pattern}", "correct": true_label == 1 if true_label is not None else None
                })
            return 1, f"positive: {pattern}"

    # Priority 5: Generic cancer substring
    if "cancer" in t:
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 1, "true_label": true_label,
                "reason": "substring_cancer", "correct": true_label == 1 if true_label is not None else None
            })
        return 1, "substring_cancer"

    # Priority 6: Letter patterns
    if re.search(r'\bb\b', t):
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 1, "true_label": true_label,
                "reason": "matched_letter_b", "correct": true_label == 1 if true_label is not None else None
            })
        return 1, "matched_letter_b"

    if re.search(r'\ba\b', t):
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 0, "true_label": true_label,
                "reason": "matched_letter_a", "correct": true_label == 0 if true_label is not None else None
            })
        return 0, "matched_letter_a"

    # Failed to parse
    if debug_log is not None:
        debug_log.append({
            "raw_output": raw_text, "normalized": t, "predicted": -1, "true_label": true_label,
            "reason": "unparseable", "correct": None
        })
    return -1, "unparseable"


class Util:
    def __init__(self):
        self.cancer = []
        self.noncancer = []
        self.model = None
        self.base_model = None
        self.train_ds = []
        self.train_headers_num_np = []
        self.train_headers_str_np = []
        self.val_ds = []
        self.test_ds = []
        self.test_labels = []
        self.fine_tune_at = 0

    #Util.load('path/') for train val data
    def load(self, img_path, header_path, random_seed=10):
        # Check image path
        omama_dir_path = Path(img_path) / "images"
        if not os.path.exists(omama_dir_path):
            print(f"File not found: {omama_dir_path}")
            print("Please update dicom_path with a valid image file path")
            return
        else:
            print(f"Found DICOM folder: {os.path.basename(omama_dir_path)}")
            omama_folder = Path(omama_dir_path)
            IDs = sorted(str(file) for file in omama_folder.rglob("*.npz"))

        # Check metadata path
        meta_dir = Path(img_path) / "metadata"

        # Get the headers
        pkl = Path(header_path)
        mapping = pickle.load(open(pkl, "rb"))

        # Get the images, labels, and headers
        noncancer = {}
        cancer = {}
        for ID in IDs :
            meta = json.load(open(meta_dir / f"{Path(ID).stem}.json"))
            if meta["label"] == "Unknown" :
                continue
        
            # Load image
            img_name = ID
            # Load window values
            wc = meta["WindowCenter"]
            ww = meta["WindowWidth"]
            window_center = float(wc[0] if hasattr(wc, '__len__') else wc)
            window_width = float(ww[0] if hasattr(ww, '__len__') else ww)
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2

            # Load headers
            ds = mapping[Path(ID).stem]
            age_str = ds.PatientAge
            if age_str and age_str != '':
                age = float(age_str[:-1]) / 12.0
            else:
                age = -1.0 
            breastimplant = ds.get("BreastImplantPresent")
            if breastimplant == "None" :
                breastimplant = "NO"
            view = ds.ViewPosition
            if meta["label"] == "NonCancer" :
                if meta["PatientID"] not in noncancer.keys() :
                    noncancer[meta["PatientID"]] = []
                noncancer[meta["PatientID"]].append([img_name, [window_min, window_max], 0, 
                                                     [age, breastimplant, view]])
            else : 
                if meta["PatientID"] not in cancer.keys() :
                    cancer[meta["PatientID"]] = []
                cancer[meta["PatientID"]].append([img_name, [window_min, window_max], 1, 
                                                  [age, breastimplant, view]])
        self.noncancer = list(noncancer.items())
        self.cancer = list(cancer.items())

        # Get the random sample of fix seed
        self.random_seed = random_seed
        split_rng = random.Random(self.random_seed)
        split_rng.shuffle(self.cancer)
        split_rng.shuffle(self.noncancer) 

    #Util.load_test('path/') for test data
    def load_test(self, use_meta=True) :  
        # Select cancer patients for test data
        test_amount = int(0.25 * len(self.cancer))
        test_c = self.cancer[:test_amount]
        test_c_data = self._assign_data(test_c)
        
        # Get same numbers of images for non-cancer cases
        test_nc = self.noncancer[:test_amount]
        test_nc_data = self._assign_data(test_nc)
        
        # Merge cancer and non-cancer, shuffle
        test_all = test_c_data + test_nc_data
        random.Random(self.random_seed).shuffle(test_all)

        test_imgs, test_metadata, test_labels, test_header_num, test_header_str = self._np_setup(test_all)
        self.test_labels = test_labels
        if use_meta :
            test_ds = tf.data.Dataset.from_tensor_slices((test_imgs, test_metadata, test_labels, test_header_num, test_header_str))
        else :
            test_ds = tf.data.Dataset.from_tensor_slices((test_imgs, test_metadata, test_labels))
        test_ds = test_ds.map(self._load_image_fn(use_meta), num_parallel_calls=tf.data.AUTOTUNE)
        self.test_ds = test_ds.batch(32)
        
    #Util.sample_data to randomly sample train/val data from large collection
    def sample_data(self, use_meta=True) :
        # Select cancer patients for train/val data
        test_amount = int(0.25 * len(self.cancer))
        trainval_c = self.cancer[test_amount:]
        trainval_c_data = self._assign_data(trainval_c)
        # Get same numbers of images for non-cancer cases
        trainval_nc = self.noncancer[test_amount:]
        trainval_nc_data = self._assign_data(trainval_nc)

        cpool_imgs, cpool_metadata, cpool_labels, cpool_header_num, cpool_header_str = self._np_setup(trainval_c_data)
        ncpool_imgs, ncpool_metadata, ncpool_labels, ncpool_header_num, ncpool_header_str = self._np_setup(trainval_nc_data)

        # Create the dataset
        if use_meta :
            c_ds = tf.data.Dataset.from_tensor_slices((cpool_imgs, cpool_metadata, cpool_labels, cpool_header_num, cpool_header_str))
            nc_ds = tf.data.Dataset.from_tensor_slices((ncpool_imgs, ncpool_metadata, ncpool_labels, ncpool_header_num, ncpool_header_str))
        else :
            c_ds = tf.data.Dataset.from_tensor_slices((cpool_imgs, cpool_metadata, cpool_labels))
            nc_ds = tf.data.Dataset.from_tensor_slices((ncpool_imgs, ncpool_metadata, ncpool_labels))
        # Take the rest of cancer pool
        c_ds = c_ds.shuffle(1000)
        n_images = int((2/3) * len(cpool_imgs))
        ctrain_ds = c_ds.take(n_images).cache()
        cval_ds = c_ds.skip(n_images).cache()
        # Take unique set of images from non-cancer pool
        nc_ds = nc_ds.shuffle(len(ncpool_imgs)).take(len(cpool_imgs))
        nctrain_ds = nc_ds.take(n_images).cache()
        ncval_ds = nc_ds.skip(n_images).cache()

        # Check balance before map
        raw_train = ctrain_ds.concatenate(nctrain_ds)
        if use_meta :
            train_labels_list = [l.numpy() for _, _, l, _, _ in raw_train]
        else :
            train_labels_list = [l.numpy() for _, _, l in raw_train]
        train_labels_arr = np.array(train_labels_list).astype(int)
        neg, pos = np.bincount(train_labels_arr)
        total = neg + pos
        print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
            total, pos, 100 * pos / total))

        # Get the headers from the train dataset for normalizing and encoding later
        if use_meta :
            train_headers_num = [h.numpy() for _, _, _, h, _ in raw_train]
            train_headers_str = [h.numpy() for _, _, _, _, h in raw_train]
            self.train_headers_num_np = np.array(train_headers_num)
            self.train_headers_str_np = np.array(train_headers_str)
        
        # Merge and shuffle
        train_ds = ctrain_ds.concatenate(nctrain_ds)
        val_ds = cval_ds.concatenate(ncval_ds)
        train_size = n_images * 2
        train_ds = train_ds.shuffle(train_size)
        train_ds = train_ds.map(self._load_image_fn(use_meta), num_parallel_calls=tf.data.AUTOTUNE)
        self.train_ds = train_ds.batch(32)
        val_ds = val_ds.shuffle(train_size)
        val_ds = val_ds.map(self._load_image_fn(use_meta), num_parallel_calls=tf.data.AUTOTUNE)
        self.val_ds = val_ds.batch(32)

    #Util.setupResnet()
    def setupResnet(self, use_meta=True) :
        # Configure dataset performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(buffer_size=AUTOTUNE)

        # Import pretrained model
        self.base_model = tf.keras.applications.ResNet50(input_shape=(1024, 1024, 3),
                                                  include_top=False, # Don't include ImageNet classifier at the top,
                                                  weights='imagenet',
                                                 )
        # Freeze base model
        self.base_model.trainable = False
        self.base_model.summary()
        self.fine_tune_at = len(self.base_model.layers) - 50
        self.model = self._build_model(0.1, use_meta, self.train_headers_num_np, self.train_headers_str_np, resnet_preprocess)
        
        # Show input pipeline and summary
        tf.keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
        self.model.summary()

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.AUC(curve="ROC", name = 'auc'),
            ]
        )


    #Util.setupEfficientNet()
    def setupEfficientNet(self, use_meta=True) :
        # Configure dataset performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(buffer_size=AUTOTUNE)
    
        # Import pretrained model
        self.base_model = tf.keras.applications.EfficientNetB0(input_shape=(1024, 1024, 3),
                                                               include_top=False, # Don't include ImageNet classifier at the top,
                                                               weights='imagenet',
        )
        # Freeze base model
        self.base_model.trainable = False
        self.base_model.summary()
        self.fine_tune_at = len(self.base_model.layers) - 20
        self.model = self._build_model(0.2, use_meta, self.train_headers_num_np, self.train_headers_str_np, lambda x : x)
            
        # Show input pipeline and summary
        tf.keras.utils.plot_model(self.model, "multi_input_and_output_model.png", show_shapes=True)
        self.model.summary()

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.AUC(curve="ROC", name = 'auc'),
            ]
        )

    def _tf_batch_to_examples(self, tf_ds, use_meta=True):
        """
        Convert a batched tf.data.Dataset (image [+ headers], label) into a
        flat list of {"image": PIL.Image, "headers": dict|None, "label": int}.
        Shared by setupMedGemma() and trainMedGemmaLoRA() so you only ever
        write this unpacking logic once.
        """
        from PIL import Image
        examples = []
        for batch in tf_ds:
            inputs, labels = batch
            img_batch = inputs[0]
 
            if use_meta:
                age_batch = inputs[1]
                implant_batch = inputs[2]
                view_batch = inputs[3]
 
            for i in range(img_batch.shape[0]):
                img_arr = img_batch[i].numpy()
                img_arr = np.repeat(img_arr, 3, axis=-1).astype(np.uint8)
                pil_img = Image.fromarray(img_arr)
 
                headers = None
                if use_meta:
                    age_val = age_batch[i].numpy().item()
                    implant_val = implant_batch[i].numpy()
                    view_val = view_batch[i].numpy()
                    implant_val = implant_val.decode() if isinstance(implant_val, bytes) else implant_val
                    view_val = view_val.decode() if isinstance(view_val, bytes) else view_val
                    headers = {"age": age_val, "breast_implant": implant_val, "view": view_val}
 
                examples.append({
                    "image": pil_img,
                    "headers": headers,
                    "label": int(labels[i].numpy()),
                })
        return examples

    # ---------- MULTIPLE PROMPT STRATEGIES ----------
    PROMPTS = {
         "original": """Classify this mammogram.
        A: NonCancer
        B: Cancer""",

            "constrained": """You are an expert radiologist analyzing a mammogram.

        Classify this image as ONE of the following:
        A: NonCancer
        B: Cancer

        Respond with ONLY the letter and label (e.g., "A: NonCancer" or "B: Cancer").
        Do NOT add explanations or additional text.

        Your classification:""",

            "few_shot": """Classify this mammogram image.

        Examples of correct responses:
        - If normal tissue: "A: NonCancer"
        - If suspicious mass detected: "B: Cancer"
        - If calcifications concerning: "B: Cancer"
        - If dense but normal: "A: NonCancer"

        Your classification for this image:""",

            "role_based": """You are an expert radiologist with 20 years of experience in mammography screening.

        Task: Classify this mammogram as NonCancer or Cancer.

        Instructions:
        1. Analyze the image for masses, calcifications, architectural distortion
        2. Respond with EXACTLY: "A: NonCancer" or "B: Cancer"
        3. Do not include explanations

        Classification:""",

            "ultra_strict": """Classify this mammogram.

        Answer format (choose one):
        A: NonCancer
        B: Cancer

        Answer:""",

            "medical_context": """Medical context: You are reviewing a screening mammogram.

        Question: Does this mammogram show signs of malignancy?

        Options:
        A: NonCancer (benign/normal findings)
        B: Cancer (malignant/suspicious findings)

        Your answer (format "X: Label"):""",

            "binary_choice": """Analyze this mammogram and select the correct classification:

        [ ] A: NonCancer
        [ ] B: Cancer

        Selected (format "X: Label"):"""
        }

    def _build_prompt_text(self, headers, use_meta):
        header_text = ""
        if use_meta and headers is not None:
            header_text = (
                f"Patient age: {headers['age']:.1f} years\n"
                f"Breast implant present: {headers['breast_implant']}\n"
                f"View position: {headers['view']}\n\n"
            )
        return f"{header_text}Classify this mammogram.\nA: NonCancer\nB: Cancer"
 
    def _print_distribution(self, name, examples):
        dist = Counter(ex["label"] for ex in examples)
        total = len(examples)
        class_names = ["NonCancer", "Cancer"]
        print(f"{name} ({total:,} samples):")
        for label_idx in sorted(dist):
            count = dist[label_idx]
            print(f"  {class_names[label_idx]}: {count:,} ({count/total*100:.1f}%)")
 
    def trainMedGemmaLoRA(self, use_meta=True,
                           output_dir="/hpcstor6/scratch01/a/anya.tongprasith001/medgemma-lora-mydata",
                           use_4bit=True,
                           num_train_epochs=8):
        """
        LoRA fine-tune MedGemma directly on self.train_ds / self.val_ds
        (already loaded via self.load()) - no need to reprocess data.
        Train/val are already balanced 50/50 by sample_data(), so no
        rebalancing is done here.
        """
        import torch
        from datasets import Dataset
        from transformers import (
            AutoProcessor,
            AutoModelForImageTextToText,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig
        from trl import SFTTrainer, SFTConfig
 
        MODEL_ID = "google/medgemma-4b-it"
        CLASS_NAMES = ["NonCancer", "Cancer"]
 
        # ---- Reuse already-loaded train_ds/val_ds, convert once ----
        train_examples = self._tf_batch_to_examples(self.train_ds, use_meta=use_meta)
        val_examples = self._tf_batch_to_examples(self.val_ds, use_meta=use_meta)
 
        print("=" * 60)
        print("DATASET DISTRIBUTION")
        print("=" * 60)
        self._print_distribution("Train", train_examples)
        self._print_distribution("Val", val_examples)
        print("=" * 60 + "\n")
 
        def to_hf_dataset(examples):
            records = []
            for ex in examples:
                answer = "A: NonCancer" if ex["label"] == 0 else "B: Cancer"
                prompt_text = self._build_prompt_text(ex["headers"], use_meta)
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
                records.append({"image": ex["image"], "messages": messages, "label": ex["label"]})
            return Dataset.from_list(records)
 
        hf_train_ds = to_hf_dataset(train_examples)
        hf_eval_ds = to_hf_dataset(val_examples)
 
        # ---- Model / processor ----
        major_cc = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        if major_cc < 8:
            raise ValueError("Need a BF16-capable GPU (Ampere+).")
 
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
 
        model_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
 
        print(f"Loading {MODEL_ID} (4bit={use_4bit})")
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "right"
 
        model.config.id2label = {i: c for i, c in enumerate(CLASS_NAMES)}
        model.config.label2id = {c: i for i, c in enumerate(CLASS_NAMES)}
        model.config.num_labels = len(CLASS_NAMES)

        # ---- LoRA config ----
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            #modules_to_save=["lm_head", "embed_tokens"],
        )

        #  ---- Data collator ----
        def collate_fn(examples):
            texts, images = [], []
            for ex in examples:
                images.append([ex["image"]])
                txt = processor.apply_chat_template(
                    ex["messages"], add_generation_prompt=False, tokenize=False
                ).strip()
                texts.append(txt)
 
            batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
 
            labels = batch["input_ids"].clone()
 
            pad_id = processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100
 
            special = processor.tokenizer.special_tokens_map
            img_token_ids = set()
            for k in ("boi_token", "eoi_token", "image_token"):
                tok = special.get(k, None)
                if tok is not None:
                    tid = processor.tokenizer.convert_tokens_to_ids(tok)
                    if tid is not None and tid != processor.tokenizer.unk_token_id:
                        img_token_ids.add(tid)
            img_token_ids.add(262144)  # MedGemma's known image placeholder token id
 
            for tid in img_token_ids:
                labels[labels == tid] = -100
 
            batch["labels"] = labels
            return batch

        #  ---- Training args ----
        args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit",
            learning_rate=2e-4,
            warmup_ratio=0.03,
            max_grad_norm=0.3,
            lr_scheduler_type="linear",
            bf16=True,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=500,
            report_to="none",
            save_strategy="steps",
            save_total_limit=3,   # keep only the latest checkpoint to save disk space
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            label_names=["labels"],
            push_to_hub=False,
            save_steps=500
        )

        #  ---- Trainer ----
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=hf_train_ds,
            eval_dataset=hf_eval_ds,
            peft_config=peft_config,
            processing_class=processor,
            data_collator=collate_fn,
        )

        #  ---- Training ----
        print("Starting training...\n")
        trainer.train()
 
        # Save the raw LoRA adapter (small, for archival/inspection)
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        print(f"\n✓ Done. Output in : {output_dir}")
 
        # Merge LoRA weights into the base model and save a full, standalone
        # checkpoint so it can be loaded directly via pipeline() later,
        # the same way as any other model repo.
        merged_dir = f"{output_dir}-merged"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        print(f"✓ Merged model saved to: {merged_dir}")
 
        if cleanup_checkpoints:
            import shutil
            import glob as glob_module
            ckpt_dirs = glob_module.glob(os.path.join(output_dir, "checkpoint-*"))
            for ckpt_dir in ckpt_dirs:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                print(f"🗑️  Removed intermediate checkpoint: {ckpt_dir}")
 
        self.medgemma_finetuned_path = merged_dir
        return trainer
 
    def setupMedGemma(self, use_meta=True, model_id="edziocodes/medgemma-breast-cancer") :
        from transformers import pipeline
        import torch
 
        # Check if model is already loaded to avoid OOM errors
        if not hasattr(self, "pipe") or getattr(self, "_pipe_model_id", None) != model_id:
            self.pipe = pipeline(
                "image-text-to-text",
                model=model_id,
                torch_dtype=torch.bfloat16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                token=os.environ.get("HF_TOKEN"),  # uses cached `hf auth login` token if unset
            )
            self._pipe_model_id = model_id
            print(f"✓ Model loaded on: {self.pipe.device} ({model_id})")
        else:
            print(f"✓ Model already loaded on: {self.pipe.device} ({model_id})")
 
        test_examples = self._tf_batch_to_examples(self.test_ds, use_meta=use_meta)

        y_true, y_pred = [], []
        debug_log = []
        n_unparsed = 0

        for ex in test_examples:
            raw_text = self._prompt(ex["image"], self.pipe, use_meta, ex["headers"]).strip()

            pred, reason = postprocess_text_to_label_lenient(
                raw_text, true_label=ex["label"], debug_log=debug_log
            )

            if pred == -1:
                n_unparsed += 1
                continue  # skip so y_true/y_pred stay aligned

            y_pred.append(pred)
            y_true.append(ex["label"])

        if n_unparsed:
            print(f"\u26a0\ufe0f  {n_unparsed}/{len(test_examples)} outputs were unparseable and skipped.")
        self.medgemma_debug_log = debug_log  # inspect later, e.g. df = pd.DataFrame(self.medgemma_debug_log)

        self._calculate_scores(y_pred, y_true)
        return y_true, y_pred
 
    def _prompt(self, image=None, pipe=None, use_meta=True, headers=None) :
        if use_meta :
            header_text = ""
            if headers is not None:
                header_text = (
                    f"Patient age: {headers['age']:.1f} years\n"
                    f"Breast implant present: {headers['breast_implant']}\n"
                    f"View position: {headers['view']}\n\n"
                )
 
            # Ask about specific conditions
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{header_text}Classify this mammogram.\nA: NonCancer\nB: Cancer"}
                    ]
                }
            ]
            output = pipe(text=messages, max_new_tokens=500)
            return output[0]["generated_text"][-1]["content"]
 
        # Ask about specific conditions
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Classify this mammogram.\nA: NonCancer\nB: Cancer"}
                ]
            }
        ]
        output = pipe(text=messages, max_new_tokens=500)
        return output[0]["generated_text"][-1]["content"]
 

    def train(self) :
        # Early stopping callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=5,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_auc',
                save_best_only=True,
                mode='max'
            )
        ]
        
        initial_epochs = 30
        history = self.model.fit(self.train_ds, 
                  epochs=initial_epochs, 
                  validation_data = self.val_ds, 
                  callbacks = callbacks)

        self.base_model.trainable = True
        # Unfreeze only the head layers
        for layer in self.base_model.layers[:self.fine_tune_at]:
            layer.trainable = False
        # Freeze BatchNorm layers
        for layer in self.base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        self.model.summary()

        # Early stopping callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=5,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_auc',
                save_best_only=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.1, 
                patience=3)
        ]

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-5, weight_decay = 0.01),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.AUC(curve="ROC", name = 'auc'),
            ],
        )

        # Train the model
        fine_tune_epochs = 40
        initial_epoch = len(history.history['auc'])
        total_epochs = initial_epoch + fine_tune_epochs
        self.model.fit(self.train_ds, 
                  epochs=total_epochs,       
                  initial_epoch=initial_epoch,  
                  validation_data=self.val_ds,
                  callbacks=callbacks)

    def predict(self) :
        best_threshold = self._best_threshold()
        self.model.evaluate(self.test_ds)
        
        # Analyse confusion matrix
        y_prob = self.model.predict(self.test_ds).reshape(-1)
        y_pred = (y_prob >= best_threshold).astype(int)
        y_true = self.test_labels.astype(int)
        self._calculate_scores(y_pred, y_true)
    
    # Assign the images, metadata, and labels for setting up mix input
    def _assign_data(self, dataset) :
        all_patient = []
        for patient, data_list in dataset :
            for data in data_list :
                img = data[0]
                metadata = data[1]
                label = data[2]
                headers = data[3]
                all_patient.append((img, metadata, label, headers))
        return all_patient

    # Set up numpy array
    def _np_setup(self, dataset) :
        imgs, metadata, labels, headers_num, headers_str = [],[],[],[],[]
        for data in dataset :
            imgs.append(data[0])
            metadata.append(data[1])
            labels.append(data[2])
            headers_num.append([data[3][0]])
            headers_str.append(list(data[3][1:]))
        imgs_np = np.array(imgs)
        metadata_np = np.array(metadata, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.float32)
        headers_num_np = np.array(headers_num, dtype=np.float32)
        headers_str_np = np.array(headers_str, dtype=str)
        return imgs_np, metadata_np, labels_np, headers_num_np, headers_str_np

    # For numeric features
    def _get_normalization_layer(self, index, numeric):
        normalizer = layers.Normalization(axis=None)

        # one numeric column by index, keep shape (batch, 1)
        feature = np.asarray(numeric[:, index], dtype=np.float32).reshape(-1, 1)

        normalizer.adapt(feature)
        return normalizer

    # For string categoray features
    def _get_category_encoding_layer(self, index, string, dtype, max_tokens=None):
        # Create a layer that turns strings into integer indices.
        if dtype == 'string':
            lookup = layers.StringLookup(max_tokens=max_tokens)
        # Otherwise, create a layer that turns integer values into integer indices.
        else:
            lookup = layers.IntegerLookup(max_tokens=max_tokens)

        # Prepare a `tf.data.Dataset` that only yields the feature.
        feature = np.asarray(string[:, index], dtype=str).reshape(-1, 1)

        # Learn the set of possible values and assign them a fixed integer index.
        lookup.adapt(feature)

        # Encode the integer indices.
        encoder = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

        # Apply multi-hot encoding to the indices. The lambda function captures the
        # layer to use or include them in the Keras Functional model later.
        return lambda feature: encoder(lookup(feature))

    # Load in the images
    def _load_image(self, path, meta, label, header_num=None, header_str=None, use_meta=True):
        def read_npz(p, m):
            p = p.decode() # From byte to string
            img = np.load(p)['data']

            window_min = m[0]
            window_max = m[1]
            img = np.clip(img, window_min, window_max)
            img = (img.astype(np.float32) - window_min) / (window_max - window_min) * 255
            return img.astype(np.float32)
    
        img = tf.numpy_function(read_npz, [path, meta], tf.float32)
        img = tf.expand_dims(img, axis=-1)
        img.set_shape([1024, 1024, 1])

        if use_meta :
            num_features = [tf.expand_dims(header_num[i], axis=-1) for i in range(1)]
            str_features = [tf.expand_dims(header_str[i], axis=-1) for i in range(2)]
            return tuple([img] + num_features + str_features), label
        return (img,), label

    def _load_image_fn(self, use_meta):
        def wrapper(*args):
            return self._load_image(*args, use_meta=use_meta)
        return wrapper

    def _build_model(self, dropout_rate, use_meta, header_num, header_str, preprocess_fn) : 
        all_inputs = []
        encoded_features = []
        if use_meta : 
            input_col = tf.keras.Input(shape=(1,), dtype=tf.float32)
            layer = self._get_normalization_layer(0, header_num)
            all_inputs.append(input_col)
            encoded_features.append(layer(input_col))

            for i in range(header_str.shape[1]):
                input_col = tf.keras.Input(shape=(1,), dtype=tf.string) 
                layer = self._get_category_encoding_layer(i, header_str, dtype="string")
                all_inputs.append(input_col)
                encoded_features.append(layer(input_col))
        
        # Setup data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])
        
        # Image branch
        # Convert image tensor from 1 channel to 3 for the model
        image_input = tf.keras.Input(shape=(1024, 1024, 1), name = 'image_input')
        x = data_augmentation(image_input)
        x = tf.keras.layers.Concatenate()([x, x, x])
        x = preprocess_fn(x)

        # Keep base model in inference mode
        x = self.base_model(x, training = False)
        x = layers.GlobalAveragePooling2D()(x) # (batch, H, W, C) becomes (batch, C) — a single feature vector per image
        #x = layers.Dense(64, activation="relu",
        #                kernel_regularizer=regularizers.l2(1e-4))(x)
        combined = layers.Dropout(dropout_rate)(x)

        if use_meta :
            # Headers branch
            header_features = layers.concatenate(encoded_features)
            h = layers.Dense(32, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(header_features)
            h = layers.Dropout(dropout_rate)(h)

            combined = layers.concatenate([x, h])

        # add activation="sigmoid" # maps feature vectors to probability score 0-1
        outputs = layers.Dense(1, activation="sigmoid")(combined) 
        model = tf.keras.Model(inputs=[image_input] + all_inputs, outputs=outputs) # wrap all layers
        return model

    def _best_threshold(self) :
        y_val_probs = self.model.predict(self.val_ds).reshape(-1)
        y_val_true = np.concatenate([y.numpy().reshape(-1) for _, y in self.val_ds]).astype(int)

        # Try thresholds from 0.01 to 0.99
        best_threshold = 0.5
        best_f1 = 0.0

        # Test different thresholds
        for threshold in np.arange(0.1, 1.0, 0.1):
            # Convert probabilities to hard 1s and 0s based on the current threshold
            y_pred_thresholded = (y_val_probs >= threshold).astype(int)
    
            # Calculate F1 score for this threshold
            current_f1 = f1_score(y_val_true, y_pred_thresholded)
    
            # Update the best one
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold

        print(f"Best Threshold Found: {best_threshold:.2f}")
        print(f"Max Validation F1-Score: {best_f1:.4f}")
        return best_threshold

    def _calculate_scores(self, y_pred=None, y_true=None) :
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['True 0', 'True 1'])
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()
        f1 = f1_score(y_true, y_pred)
        print(f"F1 Score: {f1:.4f}")