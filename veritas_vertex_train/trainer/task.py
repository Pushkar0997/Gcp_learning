import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"
import pandas as pd
import numpy as np
import torch
import datasets as hf_datasets
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from evaluate import load as load_metric

hf_datasets.config.IN_MEMORY_MAX_SIZE = 0

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "allenai/longformer-base-4096"
GCS_BUCKET       = os.environ.get("GCS_BUCKET", "gs://veritas-ai-bucket1").rstrip("/")
OUTPUT_DIR       = os.environ.get("AIP_MODEL_DIR", "./veritas_checkpoints")
MAX_LENGTH       = 1024
BATCH_SIZE       = 1
GRAD_ACCUM       = 8
EPOCHS           = 5
LR               = 1e-5
WARMUP_RATIO     = 0.06
WEIGHT_DECAY     = 0.01

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Part 1: Load ISOT ──────────────────────────────────────────────────────
print("Loading ISOT dataset from GCS...")
df_true = pd.read_csv(f"{GCS_BUCKET}/data/isot dataset/True.csv")
df_fake = pd.read_csv(f"{GCS_BUCKET}/data/isot dataset/Fake.csv")
df_true["label"] = 0
df_fake["label"] = 1
df = pd.concat([df_true, df_fake], ignore_index=True)
df["fulltext"] = df["title"] + " " + df["text"]
df_isot = df[["fulltext", "label"]].dropna()
print(f"ISOT: {len(df_isot)} examples")

# ── Part 2: Load LIAR ──────────────────────────────────────────────────────
print("Loading LIAR dataset...")
liar = load_dataset("liar", trust_remote_code=True)

def liar_to_binary(example):
    fake_labels = ["false", "barely-true", "pants-fire"]
    label_map   = {0:"false", 1:"half-true", 2:"mostly-true",
                   3:"true", 4:"barely-true", 5:"pants-fire"}
    text  = (example["statement"] or "") + " " + (example["speaker"] or "")
    label = 1 if label_map.get(example["label"], "") in fake_labels else 0
    return {"fulltext": text.strip(), "label": label}

liar_mapped = liar["train"].map(
    liar_to_binary,
    remove_columns=liar["train"].column_names
)
print(f"LIAR: {len(liar_mapped)} examples")

# ── Part 3: Load FEVER ─────────────────────────────────────────────────────
print("Loading FEVER dataset...")
fever = load_dataset("fever", "v1.0", trust_remote_code=True)

def fever_to_binary(example):
    label_str = example.get("label", "")
    if label_str == "SUPPORTS":
        label = 0
    elif label_str == "REFUTES":
        label = 1
    else:
        return {"fulltext": "", "label": -1}
    return {"fulltext": example.get("claim", ""), "label": label}

fever_mapped = fever["train"].map(
    fever_to_binary,
    remove_columns=fever["train"].column_names
)
fever_mapped = fever_mapped.filter(lambda x: x["label"] != -1)
print(f"FEVER: {len(fever_mapped)} examples")

# ── Part 4: Combine all datasets ───────────────────────────────────────────
print("Combining all datasets...")

common_features = Features({
    "fulltext": Value("string"),
    "label": Value("int64"),
})

isot_hf = Dataset.from_pandas(df_isot.reset_index(drop=True), features=common_features)

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
print(f"Total combined: {len(combined)} examples")

split    = combined.train_test_split(test_size=0.1, seed=42)
datasets_obj = DatasetDict({"train": split["train"], "test": split["test"]})
print(datasets_obj)

# ── Part 5: Tokenize ───────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
CACHE_DIR = "/tmp/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def tokenize_function(examples):
    inputs = tokenizer(
        examples["fulltext"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    global_attention_mask = [
        [0] * MAX_LENGTH for _ in range(len(inputs["input_ids"]))
    ]
    for mask in global_attention_mask:
        mask[0] = 1
    inputs["global_attention_mask"] = global_attention_mask
    return inputs

print("Tokenizing...")
tokenized = datasets_obj.map(
    tokenize_function,
    batched=True,
    batch_size=32,
    cache_file_names={
        "train": f"{CACHE_DIR}/tokenized_train.arrow",
        "test": f"{CACHE_DIR}/tokenized_test.arrow",
    },
    num_proc=1,
)
tokenized = tokenized.remove_columns(["fulltext"])
tokenized.set_format("torch")

# ── Part 6: Model ──────────────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=2
)

# ── Part 7: Metrics ────────────────────────────────────────────────────────
accuracy_metric = load_metric("accuracy")
f1_metric       = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_metric.compute(predictions=preds, references=labels)
    f1    = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {**acc, **f1}

# ── Part 8: Training arguments ─────────────────────────────────────────────
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
    gradient_checkpointing      = True,
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1",
    lr_scheduler_type           = "cosine",
    label_smoothing_factor      = 0.1,
    save_total_limit            = 2,
    push_to_hub                 = False,
    report_to                   = "none",
    dataloader_num_workers      = 4,
    logging_steps               = 100,
)

# ── Part 9: Trainer ────────────────────────────────────────────────────────
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = tokenized["train"],
    eval_dataset    = tokenized["test"],
    tokenizer       = tokenizer,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Starting training...")
trainer.train()

# ── Part 10: Save ──────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")       