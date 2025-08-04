import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model and tokenizer
model_path = "intent_classifier_resources_ai/finetuned_phi3_dental"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Choose device (MPS, CUDA, or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model.to(device)
model.eval()

# Multi-turn prompt
prompt = """User: I want to book a cleaning.\nAssistant: Sure, when would you like to come in?\nUser: Friday at 3pm.\nAssistant: Got it! May I have your full name?\nUser: Sarah Ali.\nAssistant:"""

# Encode and generate response
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

# Decode and print result
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== Output ===")
print(response)
