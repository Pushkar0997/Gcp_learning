import subprocess
import sys
import os

# ── Self-relaunch with accelerate for proper 2-GPU training ───────────────
if os.environ.get("ACCELERATE_LAUNCHED") != "1":
    os.environ["ACCELERATE_LAUNCHED"] = "1"
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "accelerate_config.yaml"
    )
    result = subprocess.run([
        sys.executable, "-m", "accelerate.commands.launch",
        "--config_file", config_path,
        os.path.abspath(__file__),
    ], check=True)
    sys.exit(result.returncode)

# ── Environment fixes — must be before any other imports ──────────────────
os.environ["PYTHONUNBUFFERED"]               = "1"
os.environ["NCCL_P2P_DISABLE"]               = "1"
os.environ["NCCL_IB_DISABLE"]               = "1"
os.environ["TOKENIZERS_PARALLELISM"]         = "false"
os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import torch

from datasets import (
    Dataset, DatasetDict,
    load_dataset, concatenate_datasets,
    Features, Value
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from evaluate import load as load_metric

# ── Config ────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "allenai/longformer-base-4096"
GCS_BUCKET       = os.environ.get("GCS_BUCKET", "gs://veritas-ai-bucket2")
OUTPUT_DIR       = os.environ.get("AIP_MODEL_DIR", "./veritas_checkpoints")
MAX_LENGTH       = 1024
BATCH_SIZE       = 8      # per GPU
GRAD_ACCUM       = 2      # effective batch = 8 x 2 GPUs x 2 = 32
EPOCHS           = 3
LR               = 2e-5
WARMUP_RATIO     = 0.06
WEIGHT_DECAY     = 0.01
CACHE_DIR        = "/tmp/hf_cache"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Startup info ──────────────────────────────────────────────────────────
print("=" * 60, flush=True)
print("Veritas AI — Training Start", flush=True)
print(f"GPUs available    : {torch.cuda.device_count()}", flush=True)
print(f"Model             : {MODEL_CHECKPOINT}", flush=True)
print(f"Max length        : {MAX_LENGTH}", flush=True)
print(f"Batch per GPU     : {BATCH_SIZE}", flush=True)
print(f"Grad accumulation : {GRAD_ACCUM}", flush=True)
print(f"Output dir        : {OUTPUT_DIR}", flush=True)
print("=" * 60, flush=True)

# ── Part 1: ISOT ──────────────────────────────────────────────────────────
print("\n[1/5] Loading ISOT...", flush=True)
df_true = pd.read_csv(f"{GCS_BUCKET}/data/isot dataset/True.csv")
df_fake = pd.read_csv(f"{GCS_BUCKET}/data/isot dataset/Fake.csv")
df_true["label"] = 0
df_fake["label"] = 1
df = pd.concat([df_true, df_fake], ignore_index=True)
df["fulltext"] = df["title"].fillna("") + " " + df["text"].fillna("")
df_isot = df[["fulltext", "label"]].dropna()
print(f"    ISOT: {len(df_isot):,} examples", flush=True)

# ── Part 2: LIAR ──────────────────────────────────────────────────────────
print("\n[2/5] Loading LIAR...", flush=True)
liar = load_dataset("liar", trust_remote_code=True, cache_dir=CACHE_DIR)

def liar_to_binary(example):
    fake_labels = ["false", "barely-true", "pants-fire"]
    label_map   = {
        0: "false",      1: "half-true",
        2: "mostly-true", 3: "true",
        4: "barely-true", 5: "pants-fire"
    }
    text  = (example["statement"] or "") + " " + (example["speaker"] or "")
    label = 1 if label_map.get(example["label"], "") in fake_labels else 0
    return {"fulltext": text.strip(), "label": label}

liar_mapped = liar["train"].map(
    liar_to_binary,
    remove_columns=liar["train"].column_names,
)
print(f"    LIAR: {len(liar_mapped):,} examples", flush=True)

# ── Part 3: FEVER ─────────────────────────────────────────────────────────
print("\n[3/5] Loading FEVER...", flush=True)
fever = load_dataset("fever", "v1.0", trust_remote_code=True, cache_dir=CACHE_DIR)

def fever_to_binary(example):
    label_str = example.get("label", "")
    if label_str == "SUPPORTS":
        return {"fulltext": example.get("claim", ""), "label": 0}
    elif label_str == "REFUTES":
        return {"fulltext": example.get("claim", ""), "label": 1}
    return {"fulltext": "", "label": -1}

fever_mapped = fever["train"].map(
    fever_to_binary,
    remove_columns=fever["train"].column_names,
)
fever_mapped = fever_mapped.filter(lambda x: x["label"] != -1)
print(f"    FEVER: {len(fever_mapped):,} examples", flush=True)

# ── Part 4: Combine ───────────────────────────────────────────────────────
print("\n[4/5] Combining datasets...", flush=True)

common_features = Features({
    "fulltext": Value("string"),
    "label":    Value("int64"),
})

isot_hf = Dataset.from_pandas(
    df_isot.reset_index(drop=True),
    features=common_features
)
liar_mapped = liar_mapped.map(
    lambda x: {"fulltext": str(x["fulltext"]), "label": int(x["label"])}
)
liar_mapped = liar_mapped.cast(common_features)

fever_mapped = fever_mapped.map(
    lambda x: {"fulltext": str(x["fulltext"]), "label": int(x["label"])}
)
fever_mapped = fever_mapped.cast(common_features)

combined = concatenate_datasets([isot_hf, liar_mapped, fever_mapped])
combined = combined.shuffle(seed=42)
print(f"    Total: {len(combined):,} examples", flush=True)

split    = combined.train_test_split(test_size=0.1, seed=42)
datasets = DatasetDict({
    "train": split["train"],
    "test":  split["test"],
})
print(f"    Train: {len(datasets['train']):,} | Test: {len(datasets['test']):,}", flush=True)

# ── Part 5: Tokenize ──────────────────────────────────────────────────────
print("\n[5/5] Tokenizing...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    inputs = tokenizer(
        examples["fulltext"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    n    = len(inputs["input_ids"])
    mask = [[0] * MAX_LENGTH for _ in range(n)]
    for m in mask:
        m[0] = 1
    inputs["global_attention_mask"] = mask
    return inputs

tokenized = datasets.map(
    tokenize_function,
    batched=True,
    batch_size=32,
    num_proc=1,
    cache_file_names={
        "train": f"{CACHE_DIR}/tokenized_train.arrow",
        "test":  f"{CACHE_DIR}/tokenized_test.arrow",
    }
)
tokenized = tokenized.remove_columns(["fulltext"])
tokenized.set_format("torch")
print("    Tokenization complete.", flush=True)

# ── Model ─────────────────────────────────────────────────────────────────
print("\nLoading model...", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=2,
)
print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# ── Metrics ───────────────────────────────────────────────────────────────
accuracy_metric = load_metric("accuracy")
f1_metric       = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_metric.compute(predictions=preds, references=labels)
    f1    = f1_metric.compute(
        predictions=preds, references=labels, average="weighted"
    )
    return {**acc, **f1}

# ── Training args ─────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    learning_rate               = LR,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_train_epochs            = EPOCHS,
    weight_decay                = WEIGHT_DECAY,
    warmup_ratio                = WARMUP_RATIO,
    fp16                        = True,
    gradient_checkpointing      = False,
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1",
    lr_scheduler_type           = "cosine",
    label_smoothing_factor      = 0.1,
    save_total_limit            = 2,
    push_to_hub                 = False,
    report_to                   = "none",
    dataloader_num_workers      = 8,
    dataloader_pin_memory       = True,
    logging_steps               = 10,
    logging_first_step          = True,
    logging_strategy            = "steps",
    ddp_find_unused_parameters  = False,
)

# ── Trainer ───────────────────────────────────────────────────────────────
print(f"\nStarting training on {torch.cuda.device_count()} GPU(s)...", flush=True)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = tokenized["train"],
    eval_dataset    = tokenized["test"],
    tokenizer       = tokenizer,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────
print("\nSaving model...", flush=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}", flush=True)
print("=" * 60, flush=True)
print("Training complete.", flush=True)
print("=" * 60, flush=True)