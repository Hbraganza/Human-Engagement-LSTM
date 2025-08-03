import os
import pandas as pd
from moviepy import VideoFileClip
import re

# === Config ===
TEST_ROOT = "/home/crazyhjb/Individual Project/Lego_data/Train"
SEGMENT_LENGTH = 10  # seconds
LABELLED_LIST = "/home/crazyhjb/Individual Project/Labelled_data_list.txt"

def get_labelled_session_ids(label_file):
    session_ids = []
    with open(label_file, "r") as f:
        for line in f:
            match = re.match(r"(\d{6})", line)
            if match:
                session_ids.append(match.group(1))
    return session_ids

def process_session(session_id, session_path):
    video_path = os.path.join(session_path, "FC2_L.mp4")
    output_dir = os.path.join(session_path, "Labels_FC2_L")
    csv_output = os.path.join(output_dir, "segments_info.csv")

    if not os.path.exists(video_path):
        print(f"[WARNING] Video not found: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    video = VideoFileClip(video_path)
    duration = int(video.duration)
    info = []
    i = 0
    for start in range(0, duration, SEGMENT_LENGTH):
        end = min(start + SEGMENT_LENGTH, duration)
        if end - start < SEGMENT_LENGTH:
            break  # skip last segment if shorter than SEGMENT_LENGTH
        clip = video.subclipped(start, end)
        out_file = os.path.join(output_dir, f"{session_id}_seg{i:03d}.mp4")
        clip.write_videofile(out_file, codec="libx264", audio_codec="aac", logger=None)
        info.append({
            "session": session_id,
            "segment_id": f"{session_id}_seg{i:03d}",
            "start": start,
            "end": end,
            "file": out_file,
            "label": ""
        })
        i += 1

    df = pd.DataFrame(info)
    df.to_csv(csv_output, index=False)
    print(f"{len(info)} segments have been saved for {session_id}, CSV file saved to: {csv_output}")

# === Main loop for only labelled session folders ===
labelled_sessions = set(get_labelled_session_ids(LABELLED_LIST))
processed_count = 0
for folder in os.listdir(TEST_ROOT):
    if folder in labelled_sessions:
        session_path = os.path.join(TEST_ROOT, folder)
        if os.path.isdir(session_path):
            process_session(folder, session_path)
            processed_count +=1

print (f"Total sessions processed: {processed_count}")