import os
import platform
import subprocess
import pandas as pd

BASE_PATH = "/home/crazyhjb/Individual Project/Lego_data"
LABELLED_LIST_PATH = "/home/crazyhjb/Individual Project/Labelled_data_list.txt"
LABEL_FOLDERS = ["Labels_FC1_L", "Labels_FC2_L"]

def open_video_with_system_player(filepath):
    if platform.system() == "Windows":
        os.startfile(filepath)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", filepath])
    else:  # Linux
        subprocess.call(["xdg-open", filepath])

def get_train_segment_ids():
    with open(LABELLED_LIST_PATH, "r") as f:
        lines = f.readlines()
    ids = []
    for line in lines:
        if "-" in line:
            ids.append(line.split("-")[0].strip())
    return ids

def get_segment_ids(folder):
    folder_path = os.path.join(BASE_PATH, folder)
    return [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and d.isdigit() and len(d) == 6]

def label_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    updated = False
    label_map = {"l": "low", "m": "medium", "h": "high"}
    for i, row in df.iterrows():
        if pd.notna(row.get("label", None)) and str(row["label"]).strip() != "":
            print(f"Already labelled: {row.get('file', 'Unknown file')}, skipping.")
            continue
        video_path = os.path.abspath(row["file"])
        print(f"\nNow playing: {video_path}")
        open_video_with_system_player(video_path)
        input("Press (Enter) to start labeling...")
        label = input("Input (l / m / h): ").strip().lower()
        while label not in label_map:
            label = input("Please only type (l / m / h): ").strip().lower()
        df.at[i, "label"] = label_map[label]
        updated = True
        df.to_csv(csv_path, index=False)
        print(f"Label saved: {label_map[label]}")
    if not updated:
        print(f"All rows already labelled in {csv_path}")
        

def main():
    folder = input("Enter which folder to label (Train/Test/Validation): ").strip()
    if folder not in ["Train", "Test", "Validation"]:
        print("Invalid folder. Please enter Train, Test, or Validation.")
        return

    if folder == "Train":
        segment_ids = get_train_segment_ids()
    else:
        segment_ids = get_segment_ids(folder)
    i = 0
    for segment_id in segment_ids:
        i +=1
        for label_folder in LABEL_FOLDERS:
            csv_path = os.path.join(BASE_PATH, folder, segment_id, label_folder, "segments_info.csv")
            print(f"\nProcessing: {csv_path}")
            label_csv(csv_path)
        print (f"\nFinished processing segment {i}/{len(segment_ids)}: {segment_id}")

if __name__ == "__main__":
    main()