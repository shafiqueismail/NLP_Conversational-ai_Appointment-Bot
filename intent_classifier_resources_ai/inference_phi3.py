import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Force CPU usage to avoid MPS out-of-memory errors
device = torch.device("cpu")

# Load fine-tuned model and tokenizer from local directory
model_path = "finetuned_phi3_dental"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

# Move model to CPU and set to eval mode
model.to(device)
model.eval()

# Example multi-turn dental booking conversation prompt
prompt = """User: I want to book a cleaning.\nAssistant: Sure, when would you like to come in?\nUser: Friday at 3pm.\nAssistant: Got it! May I have your full name?\nUser: Sarah Ali.\nAssistant:"""

# Tokenize and generate output
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)  # Shorter output for quicker testing

# Decode and print the model's response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== Output ===")
print(response)
