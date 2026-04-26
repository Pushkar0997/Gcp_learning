import subprocess
import sys
import os

# ═══════════════════════════════════════════════════════════════════════════
# SELF-RELAUNCH via accelerate  (runs exactly once)
#
# Vertex starts the container with:  python -m trainer.task
# ACCELERATE_LAUNCHED is unset → we re-exec under "accelerate launch" which
# sets up DDP, then exits. The second invocation finds ACCELERATE_LAUNCHED=1
# and falls through to the real training code.
# ═══════════════════════════════════════════════════════════════════════════
if os.environ.get("ACCELERATE_LAUNCHED") != "1":
    env = os.environ.copy()
    env["ACCELERATE_LAUNCHED"]            = "1"
    env["AIP_MODEL_DIR"]                  = os.environ.get("AIP_MODEL_DIR", "")
    env["PYTHONUNBUFFERED"]               = "1"
    env["NCCL_P2P_DISABLE"]               = "1"
    env["NCCL_IB_DISABLE"]               = "1"
    env["TOKENIZERS_PARALLELISM"]         = "false"
    env["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "accelerate_config.yaml",
    )
    result = subprocess.run(
        [
            sys.executable, "-m", "accelerate.commands.launch",
            "--config_file", config_path,
            os.path.abspath(__file__),
        ],
        env=env,
        check=False,
    )
    sys.exit(result.returncode)


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════
os.environ.setdefault("PYTHONUNBUFFERED",               "1")
os.environ.setdefault("NCCL_P2P_DISABLE",               "1")
os.environ.setdefault("NCCL_IB_DISABLE",               "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM",         "false")
os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", "0")

sys.stdout.reconfigure(line_buffering=True)

import time
import json
import csv
import pandas as pd
import numpy as np
import torch
import gcsfs

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_from_disk,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from evaluate import load as load_metric


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
RANK = int(os.environ.get("LOCAL_RANK", "0"))
import tarfile
import shutil
from datetime import datetime

def is_main() -> bool:
    return RANK == 0

def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)

def gsutil_cp_with_retry(src: str, dst: str, max_retries: int = 3) -> tuple[bool, str]:
    """Copy src → dst via gsutil with retry logic. Returns (success, error_message)."""
    log(f"  gsutil cp (with {max_retries} retries): {src}  →  {dst}")
    
    for attempt in range(1, max_retries + 1):
        log(f"    [Attempt {attempt}/{max_retries}]")
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", src, dst],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if result.stdout:
                log(f"    ✓ Upload successful")
            return True, ""
        else:
            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
            log(f"    ✗ Failed (exit code {result.returncode})")
            if attempt < max_retries:
                wait_time = 2 ** attempt  # exponential backoff: 2s, 4s, 8s
                log(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                log(f"    Final error: {error_msg}")
    
    return False, error_msg

def compress_model(src_dir: str, tar_path: str) -> tuple[bool, str]:
    """Compress model directory to tar.gz. Returns (success, error_message)."""
    log(f"  Compressing model: {src_dir} → {tar_path}")
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(src_dir, arcname=os.path.basename(src_dir))
        size_mb = os.path.getsize(tar_path) / (1024 ** 2)
        log(f"    ✓ Compressed: {size_mb:.1f} MB")
        return True, ""
    except Exception as exc:
        error_msg = str(exc)
        log(f"    ✗ Compression failed: {error_msg}")
        return False, error_msg

def verify_model_files(model_dir: str, min_files: int = 5) -> tuple[bool, int]:
    """Verify model directory has expected files. Returns (success, file_count)."""
    try:
        files = os.listdir(model_dir)
        file_count = len(files)
        if file_count >= min_files:
            return True, file_count
        else:
            return False, file_count
    except Exception:
        return False, 0

def gsutil_verify_upload(gcs_path: str) -> tuple[bool, int, str]:
    """Verify files exist in GCS. Returns (success, file_count, error_message)."""
    result = subprocess.run(
        ["gsutil", "ls", "-r", gcs_path + "/"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        file_count = len([l for l in result.stdout.strip().split('\n') if l.strip() and not l.endswith(':')])
        return True, file_count, ""
    else:
        error = result.stderr.strip() if result.stderr else "Unknown error"
        return False, 0, error

def gcs_read_text(fs: gcsfs.GCSFileSystem, path: str):
    """Open a GCS file as a UTF-8 text stream."""
    return fs.open(path, "rt", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
MODEL_CHECKPOINT = "allenai/longformer-base-4096"

GCS_BUCKET    = os.environ.get("GCS_BUCKET",    "gs://veritas-ai-bucket2")
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "")

# Strip trailing slash from AIP_MODEL_DIR if present (Vertex sometimes adds it)
if AIP_MODEL_DIR.endswith("/"):
    AIP_MODEL_DIR = AIP_MODEL_DIR.rstrip("/")

LOCAL_OUTPUT        = "/tmp/veritas_model"
LOCAL_BACKUP        = "/tmp/veritas_model_backup_final"  # local copy for manual recovery
GCS_MODEL_OUT       = f"{GCS_BUCKET}/models/veritas-combined-run"
# If AIP_MODEL_DIR is set, it will be the primary save location. Use a different backup path.
GCS_MODEL_BACKUP    = f"{GCS_BUCKET}/models/veritas-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
MODEL_TAR_GZ        = "/tmp/veritas_model.tar.gz"
TOKENIZED_SAVE_PATH = "/tmp/veritas_tokenized_dataset"
BARRIER_FILE        = "/tmp/tokenization_done.flag"
CACHE_DIR           = f"/tmp/hf_cache_rank{RANK}"   # per-rank — avoids Arrow lock

# ── GCS dataset paths (matches your actual bucket layout) ─────────────────
# Image 2 confirmed these exact folder names in veritas-ai-bucket2/data/
GCS_ISOT_TRUE  = f"{GCS_BUCKET}/data/isot dataset/True.csv"
GCS_ISOT_FAKE  = f"{GCS_BUCKET}/data/isot dataset/Fake.csv"
GCS_FEVER      = f"{GCS_BUCKET}/data/fever dataset/train.jsonl"
GCS_LIAR_TRAIN = f"{GCS_BUCKET}/data/liar_dataset/train.tsv"
GCS_LIAR_VALID = f"{GCS_BUCKET}/data/liar_dataset/valid.tsv"

# ── Training hyper-parameters ─────────────────────────────────────────────
MAX_LENGTH   = 1024    # Longformer max is 4096; 1024 fits in T4 16GB with fp16
BATCH_SIZE   = 8       # per GPU — reduce to 4 if OOM
GRAD_ACCUM   = 2       # effective batch = 8 × 2 × 2 GPUs = 32
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01

os.makedirs(LOCAL_OUTPUT, exist_ok=True)
os.makedirs(CACHE_DIR,    exist_ok=True)

log("=" * 65)
log("  Veritas AI — Training Job")
log("=" * 65)
log(f"  LOCAL_RANK     : {RANK}")
log(f"  GPUs visible   : {torch.cuda.device_count()}")
log(f"  Model          : {MODEL_CHECKPOINT}")
log(f"  Max length     : {MAX_LENGTH}")
log(f"  Batch / GPU    : {BATCH_SIZE}  |  Grad accum: {GRAD_ACCUM}")
log(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM * max(torch.cuda.device_count(), 1)}")
log(f"  GCS bucket     : {GCS_BUCKET}")
log(f"  ─ Save Locations (3 backups) ─")
log(f"    1. AIP_MODEL_DIR  : '{AIP_MODEL_DIR}'")
log(f"    2. GCS primary    : {GCS_MODEL_OUT}")
log(f"    3. GCS backup     : {GCS_MODEL_BACKUP}")
log(f"    4. Local backup   : {LOCAL_BACKUP}")
log("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING + TOKENIZATION — RANK 0 ONLY
#
# In DDP both ranks run this script in parallel. If both call datasets.map()
# at the same time they race to write the same Arrow cache files → lock crash.
# Fix: rank 0 does all data work, saves with save_to_disk(), writes a flag.
# All other ranks spin-wait on the flag, then load from the shared path.
# ═══════════════════════════════════════════════════════════════════════════

COMMON = Features({"fulltext": Value("string"), "label": Value("int64")})

if is_main():

    fs = gcsfs.GCSFileSystem()   # uses the container's service account automatically

    # ── 1 / 4  ISOT  ──────────────────────────────────────────────────────
    # True.csv → label 0 (real),  Fake.csv → label 1 (fake)
    # Columns in ISOT: title, text, subject, date
    log("\n[1/4] Loading ISOT from GCS ...")

    with gcs_read_text(fs, GCS_ISOT_TRUE) as f:
        df_true = pd.read_csv(f)
    with gcs_read_text(fs, GCS_ISOT_FAKE) as f:
        df_fake = pd.read_csv(f)

    df_true["label"] = 0
    df_fake["label"] = 1
    df_all = pd.concat([df_true, df_fake], ignore_index=True)
    df_all["fulltext"] = (
        df_all["title"].fillna("") + " " + df_all["text"].fillna("")
    ).str.strip()
    df_isot = (
        df_all[["fulltext", "label"]]
        .dropna()
        .loc[lambda d: d["fulltext"].str.len() > 10]
        .reset_index(drop=True)
    )
    log(f"      ISOT : {len(df_isot):,} rows")

    # ── 2 / 4  LIAR  ──────────────────────────────────────────────────────
    # train.tsv + valid.tsv  — tab-separated, no header row
    # Column order (LIAR dataset spec):
    #   0:id  1:label  2:statement  3:subjects  4:speaker  5:job_title
    #   6:state  7:party  8:barely_true_c  9:false_c  10:half_true_c
    #   11:mostly_true_c  12:pants_fire_c  13:context
    # Binary mapping: false / barely-true / pants-fire → 1, rest → 0
    log("\n[2/4] Loading LIAR from GCS ...")

    _LIAR_FAKE = {"false", "barely-true", "pants-fire"}

    def _parse_liar_tsv(gcs_path: str) -> list:
        rows = []
        with gcs_read_text(fs, gcs_path) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                label_str = row[1].strip().lower()
                statement = row[2].strip()
                speaker   = row[4].strip() if len(row) > 4 else ""
                if not statement:
                    continue
                label = 1 if label_str in _LIAR_FAKE else 0
                fulltext = (statement + " " + speaker).strip()
                rows.append({"fulltext": fulltext, "label": label})
        return rows

    liar_rows = _parse_liar_tsv(GCS_LIAR_TRAIN) + _parse_liar_tsv(GCS_LIAR_VALID)
    df_liar   = pd.DataFrame(liar_rows)
    log(f"      LIAR : {len(df_liar):,} rows")

    # ── 3 / 4  FEVER  ─────────────────────────────────────────────────────
    # train.jsonl — one JSON object per line
    # Keys confirmed from your head -n 2 output:
    #   id, verifiable, label, claim, evidence
    # label values: "SUPPORTS" → 0, "REFUTES" → 1, "NOT ENOUGH INFO" → skip
    log("\n[3/4] Loading FEVER from GCS ...")

    fever_rows = []
    with gcs_read_text(fs, GCS_FEVER) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            lbl   = obj.get("label", "")
            claim = obj.get("claim", "").strip()
            if not claim:
                continue
            if lbl == "SUPPORTS":
                fever_rows.append({"fulltext": claim, "label": 0})
            elif lbl == "REFUTES":
                fever_rows.append({"fulltext": claim, "label": 1})
            # NOT ENOUGH INFO → skip (no ground truth)

    df_fever = pd.DataFrame(fever_rows)
    log(f"      FEVER: {len(df_fever):,} rows")

    # ── 4 / 4  COMBINE + TOKENIZE  ────────────────────────────────────────
    log("\n[4/4] Combining and tokenizing ...")

    def _df_to_hf(df: pd.DataFrame) -> Dataset:
        df = df[["fulltext", "label"]].dropna().reset_index(drop=True)
        df["fulltext"] = df["fulltext"].astype(str)
        df["label"]    = df["label"].astype(int)
        return Dataset.from_pandas(df, features=COMMON)

    combined = concatenate_datasets([
        _df_to_hf(df_isot),
        _df_to_hf(df_liar),
        _df_to_hf(df_fever),
    ]).shuffle(seed=42)

    split    = combined.train_test_split(test_size=0.1, seed=42)
    datasets = DatasetDict({"train": split["train"], "test": split["test"]})

    log(f"      Total : {len(combined):,}")
    log(f"      Train : {len(datasets['train']):,}")
    log(f"      Test  : {len(datasets['test']):,}")

    # Tokenize
    log("\n  Tokenizing (rank 0 only — prevents DDP Arrow lock collision) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def _tokenize(examples):
        enc = tokenizer(
            examples["fulltext"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        n     = len(enc["input_ids"])
        gmask = [[0] * MAX_LENGTH for _ in range(n)]
        for row in gmask:
            row[0] = 1      # Longformer: global attention on [CLS]
        enc["global_attention_mask"] = gmask
        return enc

    tokenized = datasets.map(
        _tokenize,
        batched=True,
        batch_size=32,
        num_proc=1,         # single process — no Arrow file-lock risk
        desc="Tokenizing",
    )
    tokenized = tokenized.remove_columns(["fulltext"])

    log(f"\n  Saving tokenized dataset → {TOKENIZED_SAVE_PATH}")
    tokenized.save_to_disk(TOKENIZED_SAVE_PATH)
    log("  Save complete.")

    # Release other ranks
    with open(BARRIER_FILE, "w") as f:
        f.write("done")
    log("  Barrier flag written — releasing other ranks.")

else:
    # Non-main ranks spin-wait for rank 0 (timeout 30 min)
    print(f"[Rank {RANK}] Waiting for rank 0 to finish data prep ...", flush=True)
    waited = 0
    while not os.path.exists(BARRIER_FILE):
        time.sleep(5)
        waited += 5
        if waited % 60 == 0:
            print(f"[Rank {RANK}] Still waiting ... ({waited}s)", flush=True)
        if waited > 1800:
            print(f"[Rank {RANK}] ERROR: timed out after 30 min.", flush=True)
            sys.exit(1)
    print(f"[Rank {RANK}] Barrier cleared.", flush=True)


# ── All ranks load the shared tokenized dataset ───────────────────────────
tokenized = load_from_disk(TOKENIZED_SAVE_PATH)
tokenized.set_format("torch")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

log(f"\n  All ranks loaded tokenized dataset.")
log(f"  Train: {len(tokenized['train']):,}  |  Test: {len(tokenized['test']):,}")


# ═══════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════
log("\nLoading model ...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=2
)
log(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════
_acc_metric = load_metric("accuracy")
_f1_metric  = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = _acc_metric.compute(predictions=preds, references=labels)
    f1    = _f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {**acc, **f1}


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════
training_args = TrainingArguments(
    output_dir                  = LOCAL_OUTPUT,
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
    gradient_checkpointing      = False,    # False avoids DDP hook conflicts
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1",
    lr_scheduler_type           = "cosine",
    label_smoothing_factor      = 0.1,
    save_total_limit            = 2,
    push_to_hub                 = False,
    report_to                   = "none",
    dataloader_num_workers      = 4,
    dataloader_pin_memory       = True,
    logging_steps               = 10,
    logging_first_step          = True,
    logging_strategy            = "steps",
    ddp_find_unused_parameters  = False,
)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINER + TRAIN
# ═══════════════════════════════════════════════════════════════════════════
log(f"\nStarting training on {torch.cuda.device_count()} GPU(s) ...")

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
log("\nTraining loop finished.")

# Destroy distributed process group to avoid NCCL warnings
try:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        log("  Distributed process group destroyed.")
except Exception as exc:
    log(f"  Warning: Failed to destroy process group: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# SAVE + UPLOAD — RANK 0 ONLY (BULLETPROOF MULTI-BACKUP STRATEGY)
#
# After 28+ hours of training, model MUST be saved. This implements:
#   1. Local save (primary)
#   2. AIP_MODEL_DIR (Vertex auto-registration)
#   3. GCS primary path (user-specified)
#   4. GCS backup path (timestamped, auto-created)
#   5. Local backup copy (manual recovery if needed)
#
# Multiple attempts with retry logic. If ANY save succeeds, model is preserved.
# ═══════════════════════════════════════════════════════════════════════════
if is_main():
    log("\n" + "=" * 70)
    log("  BULLETPROOF MODEL SAVE (Rank 0 only)")
    log("=" * 70)
    
    # ────────────────────────────────────────────────────────────────────────
    # PHASE 1: Verify local model was created
    # ────────────────────────────────────────────────────────────────────────
    log("\n  [PHASE 1] Verifying local model save...")
    trainer.save_model(LOCAL_OUTPUT)
    tokenizer.save_pretrained(LOCAL_OUTPUT)
    
    model_ok, model_count = verify_model_files(LOCAL_OUTPUT, min_files=3)
    log(f"  Local model files: {model_count}")
    
    if not model_ok:
        log("  ✗ CRITICAL: Local model is empty/corrupted!")
        sys.exit(1)
    log("  ✓ Local model verified OK")
    
    # ────────────────────────────────────────────────────────────────────────
    # PHASE 2 & 3: Smart save strategy
    # If AIP_MODEL_DIR is set, use it as PRIMARY. Otherwise use GCS_MODEL_OUT.
    # ────────────────────────────────────────────────────────────────────────
    aip_save_ok = False
    gcs_primary_ok = False
    gcs_primary_error = ""
    
    # Determine primary upload target
    if AIP_MODEL_DIR:
        # Use AIP_MODEL_DIR as primary (set by Vertex)
        primary_target = AIP_MODEL_DIR
        log(f"\n  [PHASE 2] Uploading to AIP_MODEL_DIR (Vertex primary)...")
        log(f"    Target: {primary_target}")
        primary_ok, primary_error = gsutil_cp_with_retry(
            LOCAL_OUTPUT + "/.",
            primary_target + "/" if not primary_target.endswith("/") else primary_target,
            max_retries=3
        )
        aip_save_ok = primary_ok
        gcs_primary_ok = primary_ok
        if primary_ok:
            log("    ✓ Vertex primary save successful")
        else:
            log(f"    ✗ Vertex primary save failed: {primary_error}")
    else:
        # Use GCS_MODEL_OUT as primary
        log(f"\n  [PHASE 2] Uploading to GCS primary path (3 retries)...")
        log(f"    Target: {GCS_MODEL_OUT}")
        primary_ok, primary_error = gsutil_cp_with_retry(
            LOCAL_OUTPUT + "/.",
            GCS_MODEL_OUT + "/",
            max_retries=3
        )
        gcs_primary_ok = primary_ok
        if primary_ok:
            log("    ✓ GCS primary upload successful")
        else:
            log(f"    ✗ GCS primary upload failed: {primary_error}")
    
    # ────────────────────────────────────────────────────────────────────────
    # PHASE 3: Verify primary upload
    # ────────────────────────────────────────────────────────────────────────
    if gcs_primary_ok:
        verify_target = AIP_MODEL_DIR if AIP_MODEL_DIR else GCS_MODEL_OUT
        log(f"\n  [PHASE 3] Verifying primary upload...")
        verify_ok, file_count, verify_error = gsutil_verify_upload(verify_target)
        if verify_ok:
            log(f"    ✓ GCS primary verified: {file_count} files")
        else:
            log(f"    ✗ GCS primary verification failed: {verify_error}")
            gcs_primary_ok = False
    
    # ────────────────────────────────────────────────────────────────────────
    # PHASE 4: GCS backup path (always create timestamped backup)
    # ────────────────────────────────────────────────────────────────────────
    log(f"\n  [PHASE 4] Uploading to GCS backup path (timestamped)...")
    log(f"    Target: {GCS_MODEL_BACKUP}")
    gcs_backup_ok, gcs_backup_error = gsutil_cp_with_retry(
        LOCAL_OUTPUT + "/.",
        GCS_MODEL_BACKUP + "/",
        max_retries=3
    )
    
    if gcs_backup_ok:
        log("    ✓ GCS backup upload successful")
        # Verify backup
        verify_ok, file_count, _ = gsutil_verify_upload(GCS_MODEL_BACKUP)
        if verify_ok:
            log(f"    ✓ GCS backup verified: {file_count} files")
        else:
            log(f"    ✗ GCS backup verification failed (but uploaded)")
            gcs_backup_ok = False
    else:
        log(f"    ✗ GCS backup upload failed: {gcs_backup_error}")
    
    # ────────────────────────────────────────────────────────────────────────
    # PHASE 5: Local backup copy (manual recovery)
    # ────────────────────────────────────────────────────────────────────────
    log(f"\n  [PHASE 5] Creating local backup copy...")
    log(f"    Target: {LOCAL_BACKUP}")
    try:
        if os.path.exists(LOCAL_BACKUP):
            shutil.rmtree(LOCAL_BACKUP)
        shutil.copytree(LOCAL_OUTPUT, LOCAL_BACKUP)
        log(f"    ✓ Local backup created")
        local_backup_ok = True
    except Exception as exc:
        log(f"    ✗ Local backup failed: {exc}")
        local_backup_ok = False
    
    # ────────────────────────────────────────────────────────────────────────
    # PHASE 6: FINAL STATUS & DECISION
    # ────────────────────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("  SAVE STATUS SUMMARY")
    log("=" * 70)
    log(f"    Local save                : ✓ OK ({model_count} files)")
    if AIP_MODEL_DIR:
        log(f"    AIP_MODEL_DIR (primary)   : {'✓ OK' if gcs_primary_ok else '✗ FAILED'}")
    else:
        log(f"    GCS primary path          : {'✓ OK' if gcs_primary_ok else '✗ FAILED'}")
    log(f"    GCS backup (timestamped)  : {'✓ OK' if gcs_backup_ok else '✗ FAILED'}")
    log(f"    Local backup copy         : {'✓ OK' if local_backup_ok else '✗ FAILED'}")
    log("=" * 70)
    
    # Count successful saves (primary + at least 1 backup)
    save_count = sum([
        model_ok,
        gcs_primary_ok,  # Either AIP_MODEL_DIR or GCS_MODEL_OUT
        gcs_backup_ok,
        local_backup_ok,
    ])
    
    log(f"\n  Total successful saves: {save_count}/4")
    
    if save_count >= 2:
        log("  ✓ MODEL PRESERVED: Model saved to multiple locations!")
        if AIP_MODEL_DIR and gcs_primary_ok:
            log(f"    → Primary (Vertex): {AIP_MODEL_DIR}")
        elif gcs_primary_ok:
            log(f"    → Primary (GCS): {GCS_MODEL_OUT}")
        if gcs_backup_ok:
            log(f"    → Backup (timestamped): {GCS_MODEL_BACKUP}")
        if local_backup_ok:
            log(f"    → Local recovery: {LOCAL_BACKUP}")
    elif save_count == 1:
        log("  ⚠ WARNING: Model saved to only ONE location (at risk)")
        if model_ok:
            log(f"    → Local: {LOCAL_OUTPUT}")
    else:
        log("  ✗ CRITICAL: Model save failed everywhere!")
        sys.exit(1)
    
    # Cleanup barrier file
    try:
        os.remove(BARRIER_FILE)
    except Exception:
        pass
    
    log("\n" + "=" * 70)
    log("  Veritas AI training complete.")
    log("=" * 70)

else:
    print(f"[Rank {RANK}] Training complete. Save handled by rank 0.", flush=True)