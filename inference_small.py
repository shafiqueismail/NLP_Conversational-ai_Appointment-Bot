# inference_small.py
import re, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_DIR = "./finetuned_small_dental"                 # your trained LoRA
BASE_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"               # must match what you trained
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float32

tok  = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device).eval()
model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device).eval()

SYSTEM = "You are a dental reception assistant. Always respond with JSON only—no extra text."
dialog = """User: I want to book a cleaning.
Assistant: Sure, when would you like to come in?
User: Friday at 3pm.
Assistant: Got it! May I have your full name?
User: Sarah Ali."""

prompt = f"<|system|>\n{SYSTEM}\n<|user|>\n{dialog}\n<|assistant|>\n"

inputs = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,            # deterministic
        eos_token_id=tok.eos_token_id
    )

text = tok.decode(out[0], skip_special_tokens=True)
m = re.search(r"\{.*\}", text, re.S)
if not m:
    print(text)
else:
    js = m.group()
    try:
        print(json.dumps(json.loads(js), indent=2))
    except Exception:
        print(js)
