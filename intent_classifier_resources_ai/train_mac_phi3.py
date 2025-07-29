import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# Use MPS if available (Apple Silicon), otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load dataset (expects your JSON file to be present)
dataset = load_dataset("json", data_files="intent_classifier_resources_ai/multi_turn_dental_dataset.json")

# Load tokenizer and model
model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

# Apply PEFT using LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Format dataset: merge prompt + completion
def format_example(example):
    return {"text": example["prompt"] + example["completion"]}

tokenized = dataset["train"].map(format_example)
tokenized = tokenized.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_phi3_dental",
    evaluation_strategy="no",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    report_to="none",
    fp16=False,
    bf16=True if torch.backends.mps.is_available() else False
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the final model
model.save_pretrained("finetuned_phi3_dental")
tokenizer.save_pretrained("finetuned_phi3_dental")