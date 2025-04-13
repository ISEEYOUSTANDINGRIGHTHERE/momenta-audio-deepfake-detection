import os
import pandas as pd

base_dir = "data"
splits = ["training", "validation", "testing"]
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

for split in splits:
    entries = []
    for label_name in ["real", "fake"]:
        label = 0 if label_name == "real" else 1
        folder_path = os.path.join(base_dir, split, label_name)
        for file in os.listdir(folder_path):
            if file.endswith((".wav", ".mp3")):
                entries.append({
                    "filepath": os.path.join(folder_path, file),
                    "label": label
                })

    df = pd.DataFrame(entries)
    df.to_csv(os.path.join(output_dir, f"{split}.csv"), index=False)

print("✅ Metadata CSVs saved in 'processed_data/' directory.")
