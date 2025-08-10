# train_mac_small.py
# Fine-tune a tiny instruct model on M1/M2 with MPS + LoRA (JSON extraction task)

import os, re, json
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"   # optional
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# -------------------- config --------------------
DATA_PATH  = "multi_turn_dental_dataset.json"  # use your CLEANED file
OUTPUT_DIR = "./finetuned_small_dental"

# pick ONE tiny instruct model (Qwen is faster/smaller on M1)
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_LEN    = 320       # shorter seq fits M1 nicely
EPOCHS     = 2         # 2–3 is enough for ~200 examples
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR         = 1e-4
LOG_STEPS  = 1
# ------------------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float32  # MPS fine-tunes happily in FP32

print(f"Device available? mps={torch.backends.mps.is_available()}  -> using: {device}")

# ---------- load & wrap dataset (force JSON-only answers) ----------
raw = load_dataset("json", data_files=DATA_PATH)["train"]

SYSTEM = "You are a dental reception assistant. Always respond with JSON only—no extra text."
def format_row(ex):
    prompt = ex["prompt"].strip()
    completion = ex["completion"].strip()
    # extract & re-dump JSON so the model learns clean JSON-only output
    try:
        obj = json.loads(re.search(r"\{.*\}", completion, re.S).group())
        completion_json = json.dumps(obj, ensure_ascii=False)
    except Exception:
        completion_json = completion
    text = f"<|system|>\n{SYSTEM}\n<|user|>\n{prompt}\n<|assistant|>\n{completion_json}"
    return {"text": text}

ds = raw.map(format_row, remove_columns=raw.column_names)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
# ensure a pad token exists
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "<|pad|>"})

def tok_fn(batch):
    return tok(batch["text"], max_length=MAX_LEN, truncation=True, padding="max_length")

tok_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])

# ---------- model + LoRA (wrap first, THEN move to MPS) ----------
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
base.resize_token_embeddings(len(tok))

# target modules for LLaMA/Qwen-style blocks
TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=TARGETS
)
model = get_peft_model(base, peft_cfg)

# move the PEFT model to MPS
model.to(device)
print("Using device for model:", next(model.parameters()).device)
assert str(next(model.parameters()).device).startswith("mps") or device == "cpu", "Model not on MPS!"

# ---------- trainer ----------
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    logging_steps=LOG_STEPS,
    save_steps=200,
    bf16=False,          # keep off on MPS
    fp16=False,          # keep off on MPS
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds,
    tokenizer=tok,
    data_collator=collator,
)

print("Starting training…")
trainer.train()
print("Saving adapter + tokenizer to:", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print("Done.")
