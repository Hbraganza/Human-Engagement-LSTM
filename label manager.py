import os
import csv
from collections import Counter
import matplotlib.pyplot as plt

def count_labels_in_folder(folder_path):
    label_counter = Counter()
    for session_id in os.listdir(folder_path):
        session_path = os.path.join(folder_path, session_id)
        if not os.path.isdir(session_path):
            continue
        for fc in ["FC1", "FC2"]:
            label_folder = os.path.join(session_path, f"Labels_{fc}_L")
            seg_csv = os.path.join(label_folder, "segments_info.csv")
            if os.path.exists(seg_csv):
                with open(seg_csv, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        label = row["label"].strip().lower()
                        if label in ["low", "medium", "high"]:
                            label_counter[label] += 1
    return label_counter

BASE_PATH = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(BASE_PATH, "Lego_data", "Train")
VAL_PATH = os.path.join(BASE_PATH, "Lego_data", "Validation")
TEST_PATH = os.path.join(BASE_PATH, "Lego_data", "Test")

label_names = ["low", "medium", "high"]
all_counters = {}
for name, path in [("Train", TRAIN_PATH), ("Validation", VAL_PATH), ("Test", TEST_PATH)]:
    label_counter = count_labels_in_folder(path)
    all_counters[name] = label_counter
    print(f"Label counts in all segments_info.csv files ({name} folder):")
    for label in label_names:
        print(f"  {label}: {label_counter[label]}")
    print()
    sizes = [label_counter.get(l, 0) for l in label_names]
    plt.figure()
    plt.pie(sizes, labels=label_names, autopct='%1.1f%%', startangle=140)
    plt.title(f'Label Distribution in {name} Folder')
    plt.savefig(f'label_distribution_piechart_{name.lower()}.png')
    plt.close()
# --- Combined pie chart ---
combined_counter = Counter()
for c in all_counters.values():
    for k in label_names:
        combined_counter[k] += c[k]
sizes = [combined_counter.get(l, 0) for l in label_names]
plt.figure()
plt.pie(sizes, labels=label_names, autopct='%1.1f%%', startangle=140)
plt.title('Label Distribution in All Folders')
plt.savefig('label_distribution_piechart_all.png')
plt.close()
