import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # (Optional but risky) Disable memory cap

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType

# Use MPS if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("json", data_files="multi_turn_dental_dataset.json")


# Load tokenizer and model
model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)


model.to(device)

# Optional: Inspect layer names to debug target_modules
print("Available Linear layer names in Phi-3 model:")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)

# LoRA config — correct modules for Phi-3
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["qkv_proj", "out_proj", "fc1", "fc2"]
)

# Apply LoRA
model = get_peft_model(model, peft_config)

# Format dataset: merge prompt + completion
def format_example(example):
    return {"text": example["prompt"] + example["completion"]}

tokenized = dataset["train"].map(format_example)
tokenized = tokenized.map(
    lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512),
    batched=True
)

# Training args
training_args = TrainingArguments(
    output_dir="./finetuned_phi3_dental",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_dir="./logs"
)

# Collator and callback
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"Step {state.global_step} | Loss: {logs['loss']:.4f}")

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[PrintLossCallback()]
)

trainer.train()

# Save
model.save_pretrained("finetuned_phi3_dental")
tokenizer.save_pretrained("finetuned_phi3_dental")
