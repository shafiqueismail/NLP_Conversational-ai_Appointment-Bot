import json
from glob import glob

# Paths to your dataset files
dataset_files = [
    "dataset/dental_cleaning_dataset.json",
    "dataset/cavity_filling_dataset.json",
    "dataset/tooth_extraction_dataset.json",
    "dataset/general_info_dataset.json"
]

# Load and combine all examples
all_data = []
for file in dataset_files:
    with open(file, 'r') as f:
        data = json.load(f)
        all_data.extend(data)

# Save to one unified dataset
with open("intent_classifier_resources_ai/multi_turn_dental_dataset.json", "w") as f:
    json.dump(all_data, f, indent=2)

print("Merged into: intent_classifier_resources_ai/multi_turn_dental_dataset.json")
