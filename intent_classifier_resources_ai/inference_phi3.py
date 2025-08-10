import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# === Load tokenizer and PEFT config ===
tokenizer = AutoTokenizer.from_pretrained("finetuned_phi3_dental", local_files_only=True)
peft_config = PeftConfig.from_pretrained("finetuned_phi3_dental", local_files_only=True)

# === Force full model loading on CPU, no GPU, no mixed-precision ===
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    device_map=None,  # Don't try to split across devices
    torch_dtype=torch.float32  # Use full precision for CPU
)

# === Apply PEFT adapter ===
model = PeftModel.from_pretrained(base_model, "finetuned_phi3_dental")

# === Use CPU only ===
device = torch.device("cpu")
model.to(device)
model.eval()

# === Prompt to test output ===
prompt = """User: I want to book a cleaning.
Assistant: Sure, when would you like to come in?
User: Friday at 3pm.
Assistant: Got it! May I have your full name?
User: Sarah Ali.
Assistant: Please confirm the details below in JSON format:
"""

# === Run inference ===
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        eos_token_id=tokenizer.eos_token_id
    )

# === Decode the output ===
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== Full Model Output ===")
print(response)

# === Try to extract JSON block ===
match = re.search(r'\{.*?\}', response, re.DOTALL)
if match:
    try:
        json_str = match.group()
        parsed = json.loads(json_str)
        print("\n=== Parsed JSON ===")
        print(json.dumps(parsed, indent=2))
    except Exception as e:
        print("\n⚠️ Found block but failed to parse JSON:")
        print(json_str)
        print(e)
else:
    print("\n⚠️ No JSON block found in model output.")
