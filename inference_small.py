# inference_small.py
# Robust, MPS-friendly inference for your LoRA adapter + JSON post-processing

import os, re, json, time, torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === Paths / model ===
ADAPTER_DIR = "./finetuned_small_dental"          # your trained LoRA dir
BASE_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"        # MUST match what you trained on
MAX_NEW_TOKENS = 120

# === Device / dtype ===
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float32

# === Minimal prompt (replace `dialog` with your conversation) ===
SYSTEM = "You are a dental reception assistant. Always respond with JSON only—no extra text."
dialog = """User: I want to book a cleaning.
Assistant: Sure, when would you like to come in?
User: Friday at 3pm.
Assistant: Got it! May I have your full name?
User: Sarah Ali."""
prompt = f"<|system|>\n{SYSTEM}\n<|user|>\n{dialog}\n<|assistant|>\n"

# === Helpers: parse weekday/time from the dialog and normalize JSON ===
WEEKDAY_TO_DATE = {
    "monday":    "2025-08-04",
    "tuesday":   "2025-08-05",
    "wednesday": "2025-08-06",
    "thursday":  "2025-08-07",
    "friday":    "2025-08-01",
}

def parse_weekday(p: str):
    # take the LAST weekday mentioned (most recent user intent)
    m = None
    for m in re.finditer(r'\b(?:next\s+)?(monday|tuesday|wednesday|thursday|friday)\b', p, re.I):
        pass
    return m.group(1).lower() if m else None

def parse_time_12h(p: str):
    # e.g., 3pm, 11:30 am, 09 am
    m = None
    for m in re.finditer(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', p, re.I):
        pass
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2) or 0)
    ampm = m.group(3).lower()
    if ampm == "pm" and hh != 12:
        hh += 12
    if ampm == "am" and hh == 12:
        hh = 0
    return f"{hh:02d}:{mm:02d}"

def normalize_treatment(t: str | None):
    if not t:
        return None
    s = t.lower().strip()
    if "extract" in s:
        return "tooth extraction"
    if "fill" in s or "cavity" in s:
        return "cavity"
    return "cleaning" if "clean" in s else s

def default_duration(treatment: str | None):
    if not treatment:
        return 60
    return 90 if "extraction" in treatment else 60

def apply_postfix(dialog_text: str, obj: dict):
    # force date/time from dialog if present
    wk = parse_weekday(dialog_text)
    tm = parse_time_12h(dialog_text)
    if wk and wk in WEEKDAY_TO_DATE:
        obj["date"] = WEEKDAY_TO_DATE[wk]
    if tm:
        obj["time"] = tm
    # normalize treatment & duration
    obj["treatment"] = normalize_treatment(obj.get("treatment")) or "cleaning"
    obj["duration"] = default_duration(obj.get("treatment"))
    return obj

# === Load tokenizer FROM ADAPTER (so added tokens match), model with safe attention ===
tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True, padding_side="right")

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    attn_implementation="eager",   # safer on MPS
)
base.resize_token_embeddings(len(tok))
base.config.pad_token_id = tok.pad_token_id
base.config.eos_token_id = tok.eos_token_id
base.config.use_cache = False

model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval().to(device)

print("device:", next(model.parameters()).device, "| vocab:", len(tok))

# === Tokenize & generate (with MPS-safe settings + CPU fallback if needed) ===
inputs = tok(prompt, return_tensors="pt").to(device)

def generate_safe():
    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=False,
                eos_token_id=tok.eos_token_id,
            )
        return out
    except Exception as e:
        print("⚠️ MPS generate() failed, falling back to CPU:", e)
        m_cpu = model.to("cpu").eval()
        inp_cpu = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.inference_mode():
            return m_cpu.generate(
                **inp_cpu,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=False,
                eos_token_id=tok.eos_token_id,
            )

out = generate_safe()

# === Extract JSON and apply post-processing ===
text = tok.decode(out[0], skip_special_tokens=True)
m = re.search(r"\{.*\}", text, re.S)
obj = {}
if m:
    try:
        obj = json.loads(m.group())
    except Exception:
        pass

obj = apply_postfix(dialog, obj)
print(json.dumps(obj, indent=2, ensure_ascii=False))
