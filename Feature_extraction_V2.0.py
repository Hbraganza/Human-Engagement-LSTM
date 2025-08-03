import os
import re
import numpy as np
import opensmile
import subprocess
import cv2
import zipfile
import h5py
import sys
import signal
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from annotations_parser import parse_face_at, parse_gaze_at, parse_lhand_at, parse_rhand_at, parse_body_at
import itertools
from scipy.spatial import ConvexHull

BASE_PATH = "/home/crazyhjb/Individual Project/Lego_data"
FOLDERS = ["Train", "Validation", "Test"]

FACE_SIZE = 28 * 2
LHAND_SIZE = 20 * 2
RHAND_SIZE = 20 * 2
BODY_SIZE = 10 * 2
ANNOTATION_DIM = FACE_SIZE + LHAND_SIZE + RHAND_SIZE + BODY_SIZE

LARGEST_SHAPES_PATH = os.path.join(os.path.dirname(__file__), "largest_shapes.txt")

def compute_stats(landmarks):
    # landmarks: (N, 2) array
    if landmarks is None or len(landmarks) == 0 or np.all(landmarks == 0):
        # 15 features, all zeros
        return [0.0] * 15
    arr = np.array(landmarks)
    x = arr[:, 0]
    y = arr[:, 1]
    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x), np.std(y)
    min_x, min_y = np.min(x), np.min(y)
    max_x, max_y = np.max(x), np.max(y)
    span_x, span_y = max_x - min_x, max_y - min_y
    # Interpoint distances
    if arr.shape[0] > 1:
        dists = [np.linalg.norm(arr[i] - arr[j]) for i, j in itertools.combinations(range(arr.shape[0]), 2)]
        ipd_mean = np.mean(dists)
        ipd_std = np.std(dists)
        ipd_min = np.min(dists)
        ipd_max = np.max(dists)
    else:
        ipd_mean = ipd_std = ipd_min = ipd_max = 0.0
    # Convex hull area
    try:
        hull = ConvexHull(arr)
        hull_area = hull.area
    except Exception:
        hull_area = 0.0
    # Principal axis ratio
    try:
        cov = np.cov(arr, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals[1] > 0:
            axis_ratio = eigvals[0] / eigvals[1]
        else:
            axis_ratio = 0.0
    except Exception:
        axis_ratio = 0.0
    return [mean_x, mean_y, std_x, std_y, min_x, min_y, max_x, max_y, span_x, span_y,
            ipd_mean, ipd_std, ipd_min, ipd_max, hull_area, axis_ratio][:15]

def parse_srt(srt_path):
    """Parse SRT and return list of (start, end, part) tuples in ms."""
    segments = []
    time_re = re.compile(r"(\d+):(\d+):(\d+),(\d+)")
    # Use latin-1 encoding to avoid UnicodeDecodeError
    with open(srt_path, "r", encoding="latin-1") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if re.match(r"^\d+$", lines[i].strip()):
            time_line = lines[i+1].strip()
            start_str, end_str = time_line.split(" --> ")
            def to_ms(tstr):
                h, m, s, ms = map(int, time_re.match(tstr).groups())
                return ((h*60 + m)*60 + s)*1000 + ms
            start_ms = to_ms(start_str)
            end_ms = to_ms(end_str)
            part_line = lines[i+2].strip()
            part_match = re.match(r"PART\.(\d):", part_line)
            if part_match:
                part = int(part_match.group(1))
                segments.append((start_ms, end_ms, part))
            i += 1
        i += 1
    return segments

def extract_audio(video_path, out_path):
    """Extract full audio from video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def get_video_fps(video_path):
    """Get video framerate using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def extract_hdf5_if_needed(folder_path):
    hdf5_path = os.path.join(folder_path, "annotations_raw.hdf5")
    zip_path = os.path.join(folder_path, "annotations_raw.zip")
    if os.path.exists(hdf5_path):
        return hdf5_path, False
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for name in zip_ref.namelist():
                if name.endswith('annotations_raw.hdf5'):
                    zip_ref.extract(name, folder_path)
                    return os.path.join(folder_path, name), True
        raise FileNotFoundError(f"annotations_raw.hdf5 not found in {zip_path}")
    raise FileNotFoundError(f"annotations_raw.hdf5 or annotations_raw.zip not found in {folder_path}")

def extract_annotation_features(hdf5_path, num_frames):
    features = np.zeros((num_frames, ANNOTATION_DIM), dtype=np.float32)
    with h5py.File(hdf5_path, 'r') as f:
        for i in range(num_frames):
            key = f"{i:05d}"
            if key not in f:
                features[i] = np.zeros(ANNOTATION_DIM, dtype=np.float32)
                continue
            vec = []
            # Face
            _, face, _ = parse_face_at(hdf5_path, i, all=True) if key in f else (None, None, None)
            if face is not None and face.shape[0] == 28:
                vec.extend(face.flatten())
            else:
                vec.extend([0.0] * FACE_SIZE)
            # Left hand
            _, lhand, _ = parse_lhand_at(hdf5_path, i, all=True) if key in f else (None, None, None)
            if lhand is not None and lhand.shape[0] == 20:
                vec.extend(lhand.flatten())
            else:
                vec.extend([0.0] * LHAND_SIZE)
            # Right hand
            _, rhand, _ = parse_rhand_at(hdf5_path, i, all=True) if key in f else (None, None, None)
            if rhand is not None and rhand.shape[0] == 20:
                vec.extend(rhand.flatten())
            else:
                vec.extend([0.0] * RHAND_SIZE)
            # Body
            _, body, _ = parse_body_at(hdf5_path, i, all=True) if key in f else (None, None, None)
            if body is not None and body.shape[0] == 10:
                vec.extend(body.flatten())
            else:
                vec.extend([0.0] * BODY_SIZE)
            features[i] = np.array(vec, dtype=np.float32)
    return features

def get_largest_shapes():
    # Check for cached file
    if os.path.exists(LARGEST_SHAPES_PATH):
        with open(LARGEST_SHAPES_PATH, "r") as f:
            largest = json.load(f)
        # Convert to tuple for each part
        for k in largest:
            largest[k] = tuple(largest[k])
        return largest
    # Compute largest shapes
    largest = {
        "face": (0, 2),
        "lhand": (0, 2),
        "rhand": (0, 2),
        "body": (0, 2)
    }
    for folder in FOLDERS:
        folder_path = os.path.join(BASE_PATH, folder)
        if not os.path.exists(folder_path):
            continue
        for session_id in os.listdir(folder_path):
            session_path = os.path.join(folder_path, session_id)
            if not os.path.isdir(session_path):
                continue
            for part_num in [1, 2]:
                ann_folder = os.path.join(session_path, f"FC{part_num}_L")
                try:
                    hdf5_path, _ = extract_hdf5_if_needed(ann_folder)
                except Exception:
                    continue
                with h5py.File(hdf5_path, 'r') as f:
                    for key in f.keys():
                        grp = f[key]
                        # Face
                        try:
                            arr = grp["face"]["landmarks"][()]
                            arr = arr[:, :2]  # drop z/confidence
                            if arr.shape[0] > largest["face"][0]:
                                largest["face"] = (arr.shape[0], 2)
                        except Exception:
                            pass
                        # Left hand
                        try:
                            arr = grp["hands"]["left"]["landmarks"][()]
                            arr = arr[:, :2]
                            if arr.shape[0] > largest["lhand"][0]:
                                largest["lhand"] = (arr.shape[0], 2)
                        except Exception:
                            pass
                        # Right hand
                        try:
                            arr = grp["hands"]["right"]["landmarks"][()]
                            arr = arr[:, :2]
                            if arr.shape[0] > largest["rhand"][0]:
                                largest["rhand"] = (arr.shape[0], 2)
                        except Exception:
                            pass
                        # Body
                        try:
                            arr = grp["body"]["landmarks"][()]
                            arr = arr[:, :2]
                            if arr.shape[0] > largest["body"][0]:
                                largest["body"] = (arr.shape[0], 2)
                        except Exception:
                            pass
    # Save to file
    with open(LARGEST_SHAPES_PATH, "w") as f:
        json.dump(largest, f)
    return largest

def pad_or_truncate(arr, target_rows, target_cols):
    arr = np.array(arr)
    out = np.zeros((target_rows, target_cols), dtype=np.float32)
    rows = min(arr.shape[0], target_rows)
    cols = min(arr.shape[1], target_cols)
    if rows > 0 and cols > 0:
        out[:rows, :cols] = arr[:rows, :cols]
    return out

def compute_stats_no_zeros(landmarks):
    arr = np.array(landmarks)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return [0.0] * 15
    # Remove zero rows
    arr = arr[~np.all(arr == 0, axis=1)]
    if arr.shape[0] == 0:
        return [0.0] * 15
    x = arr[:, 0]
    y = arr[:, 1]
    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x), np.std(y)
    min_x, min_y = np.min(x), np.min(y)
    max_x, max_y = np.max(x), np.max(y)
    span_x, span_y = max_x - min_x, max_y - min_y
    if arr.shape[0] > 1:
        dists = [np.linalg.norm(arr[i] - arr[j]) for i, j in itertools.combinations(range(arr.shape[0]), 2)]
        ipd_mean = np.mean(dists)
        ipd_std = np.std(dists)
        ipd_min = np.min(dists)
        ipd_max = np.max(dists)
    else:
        ipd_mean = ipd_std = ipd_min = ipd_max = 0.0
    try:
        hull = ConvexHull(arr)
        hull_area = hull.area
    except Exception:
        hull_area = 0.0
    try:
        cov = np.cov(arr, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals[1] > 0:
            axis_ratio = eigvals[0] / eigvals[1]
        else:
            axis_ratio = 0.0
    except Exception:
        axis_ratio = 0.0
    return [mean_x, mean_y, std_x, std_y, min_x, min_y, max_x, max_y, span_x, span_y,
            ipd_mean, ipd_std, ipd_min, ipd_max, hull_area, axis_ratio][:15]

def average_audio_frames(feats_arr, audio_times, video_times, window_size=5):
    # window_size: number of audio frames to average around each video frame
    audio_features = np.zeros((len(video_times), feats_arr.shape[1]), dtype=np.float32)
    for i, vt in enumerate(video_times):
        idx = np.argmin(np.abs(audio_times - vt))
        start = max(0, idx - window_size//2)
        end = min(feats_arr.shape[0], idx + window_size//2 + 1)
        audio_features[i] = np.mean(feats_arr[start:end], axis=0)
    return audio_features

def pad_missing_annotations(ann_data, max_gap_frames, zero_pad_shape):
    # ann_data: list of (frame_idx, arr) for frames with data
    # zero_pad_shape: shape to pad with zeros
    # Returns: dict of frame_idx -> arr (with gaps filled)
    filled = {}
    indices = [idx for idx, _ in ann_data]
    arrs = [arr for _, arr in ann_data]
    for i in range(len(indices)-1):
        start_idx, end_idx = indices[i], indices[i+1]
        filled[start_idx] = arrs[i]
        gap = end_idx - start_idx - 1
        if gap > 0:
            if gap <= max_gap_frames:
                avg_arr = (arrs[i] + arrs[i+1]) / 2.0
                for g in range(1, gap+1):
                    filled[start_idx+g] = avg_arr.copy()
            else:
                for g in range(1, gap+1):
                    filled[start_idx+g] = np.zeros(zero_pad_shape, dtype=np.float32)
    filled[indices[-1]] = arrs[-1]
    return filled

def process_video(session_path, video_file, segments, part_num, output_npy, largest_shapes):
    video_path = os.path.join(session_path, video_file)
    audio_path = os.path.join(session_path, f"temp_{part_num}_full.wav")
    extract_audio(video_path, audio_path)
    fps = get_video_fps(video_path)
    cap = cv2.VideoCapture(video_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_shift_ms = 10
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        options={'frameShift': str(frame_shift_ms), 'frameMode': 'fixed'}
    )
    feats = smile.process_file(audio_path)
    feats_arr = feats.values
    audio_frames = feats_arr.shape[0]
    audio_dim = feats_arr.shape[1]
    mask = np.zeros(video_frames, dtype=bool)
    for start_ms, end_ms, part in segments:
        if part != part_num:
            continue
        start_frame = int((start_ms / 1000) * fps)
        end_frame = int((end_ms / 1000) * fps)
        mask[start_frame:end_frame] = True
    audio_times = np.arange(audio_frames) * frame_shift_ms / 1000.0
    video_times = np.arange(video_frames) / fps
    # Average a window of audio frames around each video frame
    audio_features = average_audio_frames(feats_arr, audio_times, video_times, window_size=5)
    audio_features[~mask] = 0.0
    # --- Annotation features ---
    ann_folder = os.path.join(session_path, f"FC{part_num}_L")
    hdf5_path = None
    extracted = False
    try:
        hdf5_path, extracted = extract_hdf5_if_needed(ann_folder)
        total_ann_dim = sum([15 + (largest_shapes[p][0]*largest_shapes[p][1]) for p in ["face", "lhand", "rhand", "body"]])
        annotation_features = np.zeros((video_frames, total_ann_dim), dtype=np.float32)
        with h5py.File(hdf5_path, 'r') as f:
            for part in ["face", "lhand", "rhand", "body"]:
                part_shape = (largest_shapes[part][0], largest_shapes[part][1])
                part_dim = largest_shapes[part][0]*largest_shapes[part][1]
                # Gather available frames
                ann_data = []
                for i in range(video_frames):
                    key_i = f"{i:05d}"
                    try:
                        if part == "face":
                            arr = f[key_i]["face"]["landmarks"][()][:, :2]
                        elif part == "lhand":
                            arr = f[key_i]["hands"]["left"]["landmarks"][()][:, :2]
                        elif part == "rhand":
                            arr = f[key_i]["hands"]["right"]["landmarks"][()][:, :2]
                        elif part == "body":
                            arr = f[key_i]["body"]["landmarks"][()][:, :2]
                        arr_padded = pad_or_truncate(arr, part_shape[0], part_shape[1])
                        ann_data.append((i, arr_padded))
                    except Exception:
                        pass
                # Fill missing frames
                max_gap_frames = int(fps * 5)
                filled = pad_missing_annotations(ann_data, max_gap_frames, (part_shape[0], part_shape[1]))
                # Assign to annotation_features
                for i in range(video_frames):
                    stats = compute_stats_no_zeros(filled.get(i, np.zeros((part_shape[0], part_shape[1]), dtype=np.float32)))
                    arr_flat = filled.get(i, np.zeros((part_shape[0], part_shape[1]), dtype=np.float32)).flatten()
                    offset = 0
                    if part == "face":
                        offset = 0
                    elif part == "lhand":
                        offset = 15 + largest_shapes["face"][0]*largest_shapes["face"][1]
                    elif part == "rhand":
                        offset = 15 + largest_shapes["face"][0]*largest_shapes["face"][1] + 15 + largest_shapes["lhand"][0]*largest_shapes["lhand"][1]
                    elif part == "body":
                        offset = 15 + largest_shapes["face"][0]*largest_shapes["face"][1] + 15 + largest_shapes["lhand"][0]*largest_shapes["lhand"][1] + 15 + largest_shapes["rhand"][0]*largest_shapes["rhand"][1]
                    annotation_features[i, offset:offset+15] = stats
                    annotation_features[i, offset+15:offset+15+part_dim] = arr_flat
    except Exception as e:
        print(f"Annotation extraction failed: {e}")
        sys.exit(1)
    full_features = np.concatenate([audio_features, annotation_features], axis=1)
    np.save(output_npy, full_features)
    # Output shape and feature split info to txt
    audio_start = 0
    audio_end = audio_features.shape[1] - 1
    visual_start = audio_end + 1
    visual_end = full_features.shape[1] - 1
    with open(os.path.join(os.path.dirname(__file__), "feature_vector_shape.txt"), "w") as f:
        f.write(f"shape: {full_features.shape}\n")
        f.write(f"audio_features: {audio_start}-{audio_end}\n")
        f.write(f"visual_features: {visual_start}-{visual_end}\n")
        f.write("# audio_features: columns for audio features (OpenSMILE, eGeMAPS, etc.)\n")
        f.write("# visual_features: columns for visual/annotation features (face, hands, body, etc.)\n")
    os.remove(audio_path)
    print(f"Saved tensor to {output_npy}")
    del audio_features, annotation_features, full_features

def process_video_no_stats(session_path, video_file, segments, part_num, output_npy, largest_shapes):
    video_path = os.path.join(session_path, video_file)
    audio_path = os.path.join(session_path, f"temp_{part_num}_full.wav")
    extract_audio(video_path, audio_path)
    fps = get_video_fps(video_path)
    cap = cv2.VideoCapture(video_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_shift_ms = 10
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        options={'frameShift': str(frame_shift_ms), 'frameMode': 'fixed'}
    )
    feats = smile.process_file(audio_path)
    feats_arr = feats.values
    audio_frames = feats_arr.shape[0]
    audio_dim = feats_arr.shape[1]
    mask = np.zeros(video_frames, dtype=bool)
    for start_ms, end_ms, part in segments:
        if part != part_num:
            continue
        start_frame = int((start_ms / 1000) * fps)
        end_frame = int((end_ms / 1000) * fps)
        mask[start_frame:end_frame] = True
    audio_times = np.arange(audio_frames) * frame_shift_ms / 1000.0
    video_times = np.arange(video_frames) / fps
    audio_features = average_audio_frames(feats_arr, audio_times, video_times, window_size=5)
    audio_features[~mask] = 0.0
    # --- Annotation features: ONLY raw landmarks, no stats ---
    ann_folder = os.path.join(session_path, f"FC{part_num}_L")
    hdf5_path = None
    extracted = False
    try:
        hdf5_path, extracted = extract_hdf5_if_needed(ann_folder)
        total_ann_dim = sum([largest_shapes[p][0]*largest_shapes[p][1] for p in ["face", "lhand", "rhand", "body"]])
        annotation_features = np.zeros((video_frames, total_ann_dim), dtype=np.float32)
        with h5py.File(hdf5_path, 'r') as f:
            offset = 0
            for part in ["face", "lhand", "rhand", "body"]:
                part_shape = (largest_shapes[part][0], largest_shapes[part][1])
                part_dim = largest_shapes[part][0]*largest_shapes[part][1]
                for i in range(video_frames):
                    key_i = f"{i:05d}"
                    try:
                        if part == "face":
                            arr = f[key_i]["face"]["landmarks"][()][:, :2]
                        elif part == "lhand":
                            arr = f[key_i]["hands"]["left"]["landmarks"][()][:, :2]
                        elif part == "rhand":
                            arr = f[key_i]["hands"]["right"]["landmarks"][()][:, :2]
                        elif part == "body":
                            arr = f[key_i]["body"]["landmarks"][()][:, :2]
                        arr_padded = pad_or_truncate(arr, part_shape[0], part_shape[1])
                        annotation_features[i, offset:offset+part_dim] = arr_padded.flatten()
                    except Exception:
                        # If missing, leave as zeros
                        pass
                offset += part_dim
    except Exception as e:
        print(f"Annotation extraction failed: {e}")
        sys.exit(1)
    full_features = np.concatenate([audio_features, annotation_features], axis=1)
    np.save(output_npy, full_features)
    # Output shape and feature split info to txt
    audio_start = 0
    audio_end = audio_features.shape[1] - 1
    visual_start = audio_end + 1
    visual_end = full_features.shape[1] - 1
    with open(os.path.join(os.path.dirname(__file__), "feature_vector_shape_no_stats.txt"), "w") as f:
        f.write(f"shape: {full_features.shape}\n")
        f.write(f"audio_features: {audio_start}-{audio_end}\n")
        f.write(f"visual_features: {visual_start}-{visual_end}\n")
        f.write("# audio_features: columns for audio features (OpenSMILE, eGeMAPS, etc.)\n")
        f.write("# visual_features: columns for visual/annotation features (face, hands, body, etc.)\n")
    os.remove(audio_path)
    print(f"Saved tensor to {output_npy}")
    del audio_features, annotation_features, full_features

def save_demo_csv():
    demo_wav = "demo_temp.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=16000:cl=mono",
        "-t", "1",
        demo_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    feats = smile.process_file(demo_wav)
    feats_arr = feats.values
    feature_names = feats.columns.to_list()
    # --- Annotation feature names ---
    stat_names = ["mean_x", "mean_y", "std_x", "std_y", "min_x", "min_y", "max_x", "max_y", "span_x", "span_y",
                  "ipd_mean", "ipd_std", "ipd_min", "ipd_max", "hull_area", "axis_ratio"]
    ann_names = []
    largest_shapes = get_largest_shapes()
    for part in ["face", "lhand", "rhand", "body"]:
        ann_names += [f"{part}_{stat}" for stat in stat_names]
        part_dim = largest_shapes[part][0]*largest_shapes[part][1]
        ann_names += [f"{part}_landmark_{i}_{xy}" for i in range(largest_shapes[part][0]) for xy in ['x', 'y']]
    demo_ann = np.zeros(len(ann_names), dtype=np.float32)
    import csv
    with open("demo_full_features.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + ann_names)
        writer.writerow(list(feats_arr[0]) + list(demo_ann))
    os.remove(demo_wav)
    print("Saved demo_full_features.csv in the project folder for reference.")

def save_demo_csv_no_stats():
    demo_wav = "demo_temp.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=16000:cl=mono",
        "-t", "1",
        demo_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    feats = smile.process_file(demo_wav)
    feats_arr = feats.values
    feature_names = feats.columns.to_list()
    # --- Annotation feature names (no stats) ---
    ann_names = []
    largest_shapes = get_largest_shapes()
    for part in ["face", "lhand", "rhand", "body"]:
        part_dim = largest_shapes[part][0]*largest_shapes[part][1]
        ann_names += [f"{part}_landmark_{i}_{xy}" for i in range(largest_shapes[part][0]) for xy in ['x', 'y']]
    demo_ann = np.zeros(len(ann_names), dtype=np.float32)
    import csv
    with open("demo_full_features_no_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + ann_names)
        writer.writerow(list(feats_arr[0]) + list(demo_ann))
    os.remove(demo_wav)
    print("Saved demo_full_features_no_stats.csv in the project folder for reference.")

def signal_handler(sig, frame):
    print('KeyboardInterrupt received. Exiting.')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def main():
    largest_shapes = get_largest_shapes()
    update_all = input("Update every .npy file even if it exists? (y/n): ").strip().lower() == "y"
    for folder in FOLDERS:
        folder_path = os.path.join(BASE_PATH, folder)
        if not os.path.exists(folder_path):
            continue
        for session_id in os.listdir(folder_path):
            session_path = os.path.join(folder_path, session_id)
            if not os.path.isdir(session_path):
                continue
            print(f"\nStarting session: {session_id}")
            srt_file = os.path.join(session_path, f"{session_id}_lego.srt")
            if not os.path.exists(srt_file):
                print(f"Transcript not found: {srt_file}")
                continue
            try:
                segments = parse_srt(srt_file)
            except UnicodeDecodeError as e:
                print(f"UnicodeDecodeError in session '{session_id}' file '{srt_file}': {e}")
                continue
            for part_num, video_file in [(1, "FC1_L.mp4"), (2, "FC2_L.mp4")]:
                video_path = os.path.join(session_path, video_file)
                if not os.path.exists(video_path):
                    print(f"Video not found: {video_path}")
                    continue
                # --- New output filename ---
                output_npy = os.path.join(session_path, f"{session_id}_FC{part_num}_L_full_features_no_stats.npy")
                if os.path.exists(output_npy) and not update_all:
                    print(f"Skipping (already exists): {output_npy}")
                    continue
                process_video_no_stats(session_path, video_file, segments, part_num, output_npy, largest_shapes)

if __name__ == "__main__":
    save_demo_csv_no_stats()
    main()