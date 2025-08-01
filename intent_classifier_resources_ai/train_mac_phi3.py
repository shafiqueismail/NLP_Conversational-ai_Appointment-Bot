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
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load dataset
dataset = load_dataset("json", data_files="intent_classifier_resources_ai/multi_turn_dental_dataset.json")

# Load tokenizer and model (force 4-bit or bf16 if using MPS)
model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in lower precision to save memory
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type == "mps" else torch.float32,  # Use float16 on MPS
    low_cpu_mem_usage=True
)
model.to(device)

# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Merge prompt + completion
def format_example(example):
    return {"text": example["prompt"] + example["completion"]}

tokenized = dataset["train"].map(format_example)
tokenized = tokenized.map(
    lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512),
    batched=True
)

# Training config
training_args = TrainingArguments(
    output_dir="./finetuned_phi3_dental",
    evaluation_strategy="no",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    per_device_train_batch_size=1,  # VERY important for MPS
    num_train_epochs=3,
    report_to="none",
    logging_dir="./logs",
    disable_tqdm=False,
    fp16=False,
    bf16=torch.backends.mps.is_available()  # Only enable bf16 if MPS
)

# Collator & callback
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"📉 Step {state.global_step} | Loss: {logs['loss']:.4f}")

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[PrintLossCallback()]
)

# Train
trainer.train()

# Save
model.save_pretrained("finetuned_phi3_dental")
tokenizer.save_pretrained("finetuned_phi3_dental")
