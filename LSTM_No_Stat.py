import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =====================
# TWEAKABLE PARAMETERS FOR EACH LSTM MODEL
# =====================
MODEL_CONFIGS = {
    'combined': {
        'hidden_dim': 128,      # LSTM hidden size for combined model
        'num_layers': 2,        # Number of LSTM layers for combined model
        'dropout': 0.3,         # Dropout for combined model
        'learning_rate': 1e-4,  # Learning rate for combined model
        'epochs': 30,           # Max epochs for combined model
        'batch_size': 16,       # Batch size for combined model
    },
    'audio': {
        'hidden_dim': 64,       # LSTM hidden size for audio-only model
        'num_layers': 1,        # Number of LSTM layers for audio-only model
        'dropout': 0.2,         # Dropout for audio-only model
        'learning_rate': 5e-4,  # Learning rate for audio-only model
        'epochs': 30,           # Max epochs for audio-only model
        'batch_size': 16,       # Batch size for audio-only model
    },
    'visual': {
        'hidden_dim': 128,      # LSTM hidden size for visual-only model
        'num_layers': 2,        # Number of LSTM layers for visual-only model
        'dropout': 0.4,         # Dropout for visual-only model
        'learning_rate': 1e-4,  # Learning rate for visual-only model
        'epochs': 30,           # Max epochs for visual-only model
        'batch_size': 16,       # Batch size for visual-only model
    },
}

# Choose which models to run: any subset of ['combined', 'audio', 'visual']
RUN_MODELS = ['combined', 'audio', 'visual']
# =====================

# --- Config ---
BASE_PATH = os.path.dirname(__file__)
TEMP_PATH = os.path.join(BASE_PATH, "temp")
os.makedirs(TEMP_PATH, exist_ok=True)
TRAIN_PATH = os.path.join(BASE_PATH, "Lego_data", "Train")
VAL_PATH = os.path.join(BASE_PATH, "Lego_data", "Validation")
TEST_PATH = os.path.join(BASE_PATH, "Lego_data", "Test")
LABELLED_LIST = os.path.join(BASE_PATH, "Labelled_data_list.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tweakable LSTM Hyperparameters
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_NUM_CLASSES = 3
LSTM_LEARNING_RATE = 1e-3
LSTM_PRETRAIN_EPOCHS = 5
LSTM_PSEUDO_EPOCHS = 5
SEGMENT_LEN = 10  # seconds
LSTM_DROPOUT = 0.3  # Dropout probability
LSTM_BATCH_SIZE = 16  # Batch size for training

# =====================
# OVERSAMPLING & UNDERSAMPLING FACTORS FOR EACH LABEL
# =====================
OVERSAMPLE = {'low': 10, 'medium': 5, 'high': 1}  # e.g. {'low': 4, 'medium': 1, 'high': 1}
UNDERSAMPLE = {'low': 1.0, 'medium': 1.0, 'high': 0.2}  # e.g. 0.5 means keep half, 1.0 means keep all
# =====================

# Helper functions
def get_labelled_sessions():
    sessions = set()
    with open(LABELLED_LIST, "r") as f:
        for line in f:
            sid = line.strip()[:6]
            if sid:
                sessions.add(sid)
    return sessions

def get_segments_info_csv(session_path, fc):
    label_folder = os.path.join(session_path, f"Labels_{fc}_L")
    return os.path.join(label_folder, "segments_info.csv")

def load_segments_info(csv_path):
    segments = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append({
                "start": float(row["start"]),
                "end": float(row["end"]),
                "label": row["label"].strip().lower()
            })
    return segments

def label_to_int(label):
    return {"low": 0, "medium": 1, "high": 2}.get(label, -1)

def int_to_label(idx):
    return ["low", "medium", "high"][idx]

def get_npy_path(session_path, fc):
    return glob(os.path.join(session_path, f"*_{fc}_L_eGeMAPS.npy"))[0]

def get_video_path(session_path, fc):
    return os.path.join(session_path, f"FC{fc}_L.mp4")

def get_fps(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def segment_indices(fps, total_frames, segment_len=SEGMENT_LEN):
    seg_size = int(fps * segment_len)
    indices = []
    for start in range(0, total_frames, seg_size):
        end = start + seg_size
        if end > total_frames:
            break
        indices.append((start, end))
    return indices

# --- Model ---
class EngagementLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

# --- DataLoader ---
def load_segment_features(npy_path, indices):
    arr = np.load(npy_path, mmap_mode='r')
    segs = []
    for start, end in indices:
        seg = arr[start:end]
        segs.append(seg)
    return segs

# --- Training/Validation/Test ---
def run_epoch(model, data, optimizer=None, batch_size=LSTM_BATCH_SIZE, class_weights=None):
    criterion = FocalLoss(alpha=class_weights)
    total_loss = 0.0
    correct = 0
    total = 0
    model.train() if optimizer else model.eval()
    # Shuffle data for training
    if optimizer:
        np.random.shuffle(data)
    # Mini-batch training
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        xs = [torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0) for x, y in batch]
        ys = [y for x, y in batch]
        x = torch.cat(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.long, device=DEVICE)
        out = model(x)
        loss = criterion(out, y)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(batch)
        preds = out.argmax(dim=1).cpu().numpy()
        correct += np.sum(preds == y.cpu().numpy())
        total += len(batch)
    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return avg_loss, acc

# --- Pseudo-labelling ---
def pseudo_label(model, unlabeled_data):
    pseudo = []
    model.eval()
    for x in unlabeled_data:
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        out = model(x)
        pred = out.argmax(dim=1).item()
        pseudo.append(pred)
    return pseudo

# --- New: Read feature vector shape and set input dims ---
def get_feature_vector_shape():
    shape_file = os.path.join(os.path.dirname(__file__), "feature_vector_shape.txt")
    with open(shape_file, "r") as f:
        line = f.readline().strip()
        shape = eval(line)
    return shape  # (num_frames, num_features)

# --- New: Parse feature_vector_shape_no_stats.txt for feature splits ---
def get_feature_splits_from_shape_txt(txt_path):
    audio_start = audio_end = visual_start = visual_end = None
    total_dim = None
    with open(txt_path, 'r') as f:
        for line in f:
            if line.startswith('shape:'):
                shape = eval(line.split(':', 1)[1].strip())
                total_dim = shape[1]
            elif line.startswith('audio_features:'):
                rng = line.split(':', 1)[1].strip()
                audio_start, audio_end = map(int, rng.split('-'))
            elif line.startswith('visual_features:'):
                rng = line.split(':', 1)[1].strip()
                visual_start, visual_end = map(int, rng.split('-'))
    return audio_start, audio_end, visual_start, visual_end, total_dim

# --- New: Get audio/visual split ---
def get_audio_visual_split():
    # eGeMAPS is 88 features, rest are visual
    audio_dim = 88
    _, total_dim = get_feature_vector_shape()
    visual_dim = total_dim - audio_dim
    return audio_dim, visual_dim

# --- New: Metrics and plotting ---
def compute_metrics(y_true, y_pred, labels=[0,1,2]):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return acc, prec, rec, f1, cm

def save_metrics_to_txt(metrics, model_name, out_dir, mode=None, loss_name=None):
    acc, prec, rec, f1, cm = metrics
    fname = os.path.join(out_dir, f"{model_name}_metrics.txt")
    header = f"\n{'='*40}\nMode: {mode or 'normal'} | Loss: {loss_name or 'Focal Loss'}\n{'='*40}\n"
    with open(fname, "a") as f:
        f.write(header)
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Confusion matrix:\n{cm}\n")

def plot_accuracy_vs_epochs(acc_list, model_name, out_dir):
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(acc_list)+1), acc_list, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy vs Epochs')
    plt.savefig(os.path.join(out_dir, f"{model_name}_accuracy_vs_epochs.png"))
    plt.close()

def plot_f1_vs_epochs(f1_list, model_name, out_dir):
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(f1_list)+1), f1_list, marker='o', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{model_name} F1 Score vs Epochs')
    plt.savefig(os.path.join(out_dir, f"{model_name}_f1_vs_epochs.png"))
    plt.close()

def plot_comparison(acc_dict, out_dir):
    plt.figure(figsize=(12,8))
    for model_name, acc_list in acc_dict.items():
        plt.plot(range(1, len(acc_list)+1), acc_list, marker='o', label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.8,1])
    plt.savefig(os.path.join(out_dir, "model_accuracy_comparison.png"), bbox_inches='tight')
    plt.close()

def plot_f1_comparison(f1_dict, out_dir):
    plt.figure(figsize=(12,8))
    for model_name, f1_list in f1_dict.items():
        plt.plot(range(1, len(f1_list)+1), f1_list, marker='o', label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Model F1 Score Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.8,1])
    plt.savefig(os.path.join(out_dir, "model_f1_comparison.png"), bbox_inches='tight')
    plt.close()

# --- EngagementLSTM with flexible hyperparameters ---
class EngagementLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- New: Focal Loss implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- New: Count true label distribution from segments_info.csv files (Train folder) ---
def count_labels_in_train_segments_info():
    train_path = TRAIN_PATH
    label_counter = Counter()
    for session_id in os.listdir(train_path):
        session_path = os.path.join(train_path, session_id)
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

# --- New: Main update ---
def main():
    start_time = time.time()
    out_dir = os.path.join(os.path.dirname(__file__), "LSTM_results")
    os.makedirs(out_dir, exist_ok=True)
    labelled_sessions = get_labelled_sessions()
    label_names = ['low', 'medium', 'high']
    # --- Print and plot label counts from segments_info.csv files (true original label distribution) ---
    true_label_counter = count_labels_in_train_segments_info()
    print("Label counts in all segments_info.csv files (Train folder):")
    for label in label_names:
        print(f"  {label}: {true_label_counter[label]}")
    true_label_sizes = [true_label_counter.get(l, 0) for l in label_names]
    plt.figure()
    plt.pie(true_label_sizes, labels=label_names, autopct='%1.1f%%', startangle=140)
    plt.title('Label Distribution in Train segments_info.csv files')
    plt.savefig(os.path.join(out_dir, 'label_distribution_piechart_true_original.png'))
    plt.close()
    # --- Pre-check: Verify .npy file column counts against required total_dim ---
    print("\n[Pre-check] Verifying .npy file column counts...")
    npy_paths = []
    for split_path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not os.path.exists(split_path):
            continue
        for session_id in os.listdir(split_path):
            session_path = os.path.join(split_path, session_id)
            if not os.path.isdir(session_path):
                continue
            for fc in ["FC1", "FC2"]:
                npy_files = glob(os.path.join(session_path, f"*_{fc}_L_full_features.npy"))
                for npy_path in npy_files:
                    npy_paths.append(npy_path)
    # Get expected number of columns from feature_vector_shape.txt
    shape_txt = os.path.join(os.path.dirname(__file__), "feature_vector_shape.txt")
    _, _, _, _, expected_dim = get_feature_splits_from_shape_txt(shape_txt)
    # Scan all .npy files and collect their column counts
    mismatches = []
    for npy_path in npy_paths:
        try:
            arr = np.load(npy_path, mmap_mode='r')
            ncol = arr.shape[1]
            if ncol != expected_dim:
                change = "drops" if ncol < expected_dim else "increases"
                mismatches.append((npy_path, ncol, change))
        except Exception as e:
            mismatches.append((npy_path, f'ERROR: {e}", error'))
    if mismatches:
        print(f"[WARNING] The following .npy files do not match the required {expected_dim} columns:")
        for path, ncol, change in mismatches:
            print(f"  {path}: {ncol} columns ({change} from required)")
        print("[Pre-check failed] Please regenerate these files to match the required feature dimension. Aborting.")
        return
    else:
        print(f"All .npy files have the required {expected_dim} columns.")
    print("[Pre-check complete]\n")
    # --- Use no_stats feature split file ---
    shape_txt = os.path.join(os.path.dirname(__file__), "feature_vector_shape_no_stats.txt")
    audio_start, audio_end, visual_start, visual_end, total_dim = get_feature_splits_from_shape_txt(shape_txt)
    audio_cols = list(range(audio_start, audio_end+1))
    visual_cols = list(range(visual_start, visual_end+1))
    audio_dim = len(audio_cols)
    visual_dim = len(visual_cols)
    # --- Prompt user for model selection ---
    print("Which LSTM(s) do you want to run?")
    print("1: Combined (audio+visual)\n2: Audio only\n3: Visual only\n4: Both audio and visual\n5: All three")
    choice = input("Enter 1, 2, 3, 4, or 5: ").strip()
    if choice == '1':
        run_models = ['combined']
    elif choice == '2':
        run_models = ['audio']
    elif choice == '3':
        run_models = ['visual']
    elif choice == '4':
        run_models = ['audio', 'visual']
    elif choice == '5':
        run_models = ['combined', 'audio', 'visual']
    else:
        print("Invalid choice. Defaulting to all three.")
        run_models = ['combined', 'audio', 'visual']
    # --- Gather training data ---
    train_data = []
    for session_id in os.listdir(TRAIN_PATH):
        session_path = os.path.join(TRAIN_PATH, session_id)
        if not os.path.isdir(session_path):
            continue
        for fc in ["FC1", "FC2"]:
            npy_files = glob(os.path.join(session_path, f"*_{fc}_L_full_features_no_stats.npy"))
            if not npy_files:
                continue
            npy_path = npy_files[0]
            video_path = get_video_path(session_path, fc[-1])
            if not os.path.exists(video_path):
                continue
            fps = get_fps(video_path)
            arr = np.load(npy_path, mmap_mode='r')
            indices = segment_indices(fps, arr.shape[0], segment_len=SEGMENT_LEN)
            label_folder = os.path.join(session_path, f"Labels_{fc}_L")
            seg_csv = os.path.join(label_folder, "segments_info.csv")
            if session_id in labelled_sessions and os.path.exists(seg_csv):
                seg_info = load_segments_info(seg_csv)
                for idx, (start, end) in enumerate(indices):
                    if idx >= len(seg_info):
                        break
                    label = label_to_int(seg_info[idx]["label"])
                    if label == -1:
                        continue
                    seg = arr[start:end]
                    train_data.append((seg, label))
    # --- Print and plot label counts for original training data (before resampling) ---
    orig_label_counts = Counter([y for _, y in train_data])
    orig_label_sizes = [orig_label_counts.get(i, 0) for i in range(3)]
    print("Label counts in original training data:")
    for i, name in enumerate(label_names):
        print(f"  {name}: {orig_label_sizes[i]}")
    plt.figure()
    plt.pie(orig_label_sizes, labels=label_names, autopct='%1.1f%%', startangle=140)
    plt.title('Label Distribution in Original Training Data')
    plt.savefig(os.path.join(out_dir, 'label_distribution_piechart_original.png'))
    plt.close()
    # --- Split features ---
    def split_features(seg):
        audio = seg[:, audio_cols]
        visual = seg[:, visual_cols]
        return audio, visual
    # --- Prepare data for each model ---
    train_sets = {
        'combined': [(x, y) for x, y in train_data],
        'audio': [(x[:, audio_cols], y) for x, y in train_data],
        'visual': [(x[:, visual_cols], y) for x, y in train_data],
    }
    input_dims = {
        'combined': total_dim,
        'audio': audio_dim,
        'visual': visual_dim,
    }
    # --- Data preparation modes ---
    def prepare_train_data(mode):
        if mode == 'normal':
            return list(train_data)
        elif mode == 'oversample':
            oversampled = []
            for x, y in train_data:
                label_name = label_names[y]
                oversampled.extend([(x, y)] * OVERSAMPLE[label_name])
            return oversampled
        elif mode == 'undersample':
            from random import random
            undersampled = []
            for x, y in train_data:
                label_name = label_names[y]
                if random() < UNDERSAMPLE[label_name]:
                    undersampled.append((x, y))
            return undersampled
        else:
            raise ValueError('Unknown mode')

    modes = ['normal', 'oversample', 'undersample']
    acc_dict_all = {}
    f1_dict_all = {}
    for mode in modes:
        print(f"\n=== Running LSTM with {mode} sampling ===")
        mode_train_data = prepare_train_data(mode)
        # --- Pie chart for this mode ---
        mode_label_counts = Counter([y for _, y in mode_train_data])
        mode_label_sizes = [mode_label_counts.get(i, 0) for i in range(3)]
        plt.figure()
        plt.pie(mode_label_sizes, labels=label_names, autopct='%1.1f%%', startangle=140)
        plt.title(f'Label Distribution ({mode})')
        plt.savefig(os.path.join(out_dir, f'label_distribution_piechart_{mode}.png'))
        plt.close()
        # --- Compute class weights for focal loss ---
        total = sum(mode_label_sizes)
        class_weights = [0 if c == 0 else total / (3 * c) for c in mode_label_sizes]
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
        # --- Print label counts ---
        print(f"Label counts in training data ({mode}):")
        for i, name in enumerate(label_names):
            print(f"  {name}: {mode_label_sizes[i]}")
        # --- Prepare data for each model ---
        train_sets = {
            'combined': [(x, y) for x, y in mode_train_data],
            'audio': [(x[:, audio_cols], y) for x, y in mode_train_data],
            'visual': [(x[:, visual_cols], y) for x, y in mode_train_data],
        }
        input_dims = {
            'combined': total_dim,
            'audio': audio_dim,
            'visual': visual_dim,
        }
        results = {}
        acc_dict = {}
        f1_dict = {}
        for model_name in run_models:
            print(f"\n--- Training {model_name} LSTM ({mode}) ---")
            cfg = MODEL_CONFIGS[model_name]
            model = EngagementLSTM(
                input_dims[model_name],
                cfg['hidden_dim'],
                cfg['num_layers'],
                LSTM_NUM_CLASSES,
                cfg['dropout']
            ).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
            acc_list = []
            f1_list = []
            for epoch in range(cfg['epochs']):
                loss, acc = run_epoch(model, train_sets[model_name], optimizer, batch_size=cfg['batch_size'], class_weights=class_weights_tensor)
                # --- Validation for F1 ---
                val_data = []
                for session_id in os.listdir(VAL_PATH):
                    session_path = os.path.join(VAL_PATH, session_id)
                    if not os.path.isdir(session_path):
                        continue
                    for fc in ["FC1", "FC2"]:
                        npy_files = glob(os.path.join(session_path, f"*_{fc}_L_full_features_no_stats.npy"))
                        if not npy_files:
                            continue
                        npy_path = npy_files[0]
                        video_path = get_video_path(session_path, fc[-1])
                        if not os.path.exists(video_path):
                            continue
                        fps = get_fps(video_path)
                        arr = np.load(npy_path, mmap_mode='r')
                        indices = segment_indices(fps, arr.shape[0], segment_len=SEGMENT_LEN)
                        label_folder = os.path.join(session_path, f"Labels_{fc}_L")
                        seg_csv = os.path.join(label_folder, "segments_info.csv")
                        if os.path.exists(seg_csv):
                            seg_info = load_segments_info(seg_csv)
                            for idx, (start, end) in enumerate(indices):
                                if idx >= len(seg_info):
                                    break
                                label = label_to_int(seg_info[idx]["label"])
                                if label == -1:
                                    continue
                                seg = arr[start:end]
                                if model_name == 'combined':
                                    val_data.append((seg, label))
                                elif model_name == 'audio':
                                    val_data.append((seg[:, audio_cols], label))
                                elif model_name == 'visual':
                                    val_data.append((seg[:, visual_cols], label))
                y_true, y_pred = [], []
                for x, y in val_data:
                    x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    out = model(x_tensor)
                    pred = out.argmax(dim=1).item()
                    y_true.append(y)
                    y_pred.append(pred)
                metrics = compute_metrics(y_true, y_pred)
                acc_list.append(metrics[0])
                f1_list.append(metrics[3])
                print(f"{model_name} Epoch {epoch+1}: loss={loss:.4f}, acc={metrics[0]:.4f}, f1={metrics[3]:.4f}")
            plot_accuracy_vs_epochs(acc_list, f"{model_name}_{mode}", out_dir)
            plot_f1_vs_epochs(f1_list, f"{model_name}_{mode}", out_dir)
            acc_dict[model_name] = acc_list
            f1_dict[model_name] = f1_list
            # --- Final validation metrics ---
            val_data = []
            for session_id in os.listdir(VAL_PATH):
                session_path = os.path.join(VAL_PATH, session_id)
                if not os.path.isdir(session_path):
                    continue
                for fc in ["FC1", "FC2"]:
                    npy_files = glob(os.path.join(session_path, f"*_{fc}_L_full_features_no_stats.npy"))
                    if not npy_files:
                        continue
                    npy_path = npy_files[0]
                    video_path = get_video_path(session_path, fc[-1])
                    if not os.path.exists(video_path):
                        continue
                    fps = get_fps(video_path)
                    arr = np.load(npy_path, mmap_mode='r')
                    indices = segment_indices(fps, arr.shape[0], segment_len=SEGMENT_LEN)
                    label_folder = os.path.join(session_path, f"Labels_{fc}_L")
                    seg_csv = os.path.join(label_folder, "segments_info.csv")
                    if os.path.exists(seg_csv):
                        seg_info = load_segments_info(seg_csv)
                        for idx, (start, end) in enumerate(indices):
                            if idx >= len(seg_info):
                                break
                            label = label_to_int(seg_info[idx]["label"])
                            if label == -1:
                                continue
                            seg = arr[start:end]
                            if model_name == 'combined':
                                val_data.append((seg, label))
                            elif model_name == 'audio':
                                val_data.append((seg[:, audio_cols], label))
                            elif model_name == 'visual':
                                val_data.append((seg[:, visual_cols], label))
            # --- Evaluate ---
            y_true, y_pred = [], []
            for x, y in val_data:
                x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                out = model(x_tensor)
                pred = out.argmax(dim=1).item()
                y_true.append(y)
                y_pred.append(pred)
            metrics = compute_metrics(y_true, y_pred)
            # --- F1-score per class ---
            f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)
            # --- Accuracy per class ---
            acc_per_class = []
            for i in [0,1,2]:
                true_i = np.array(y_true) == i
                if np.sum(true_i) == 0:
                    acc_per_class.append(0.0)
                else:
                    acc_per_class.append(np.mean(np.array(y_pred)[true_i] == i))
            with open(os.path.join(out_dir, f"{model_name}_metrics.txt"), "a") as f:
                f.write(f"F1-score per class (low, medium, high): {f1_per_class}\n")
                f.write(f"Accuracy per class (low, medium, high): {acc_per_class}\n")
            print(f"F1-score per class (low, medium, high): {f1_per_class}")
            print(f"Accuracy per class (low, medium, high): {acc_per_class}")
            save_metrics_to_txt(metrics, model_name, out_dir, mode=mode, loss_name="Focal Loss")
            results[model_name] = metrics
            acc_dict[model_name] = acc_list
        acc_dict_all[mode] = acc_dict
        f1_dict_all[mode] = f1_dict
    # --- Comparison plot for all modes ---
    plt.figure(figsize=(12,8))
    for mode in modes:
        for model_name in run_models:
            acc_list = acc_dict_all[mode][model_name]
            plt.plot(range(1, len(acc_list)+1), acc_list, marker='o', label=f'{model_name} ({mode})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison (Sampling Methods)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.8,1])
    plt.savefig(os.path.join(out_dir, "model_accuracy_comparison_sampling.png"), bbox_inches='tight')
    plt.close()
    # --- F1 comparison plot for all modes ---
    plt.figure(figsize=(12,8))
    for mode in modes:
        for model_name in run_models:
            f1_list = f1_dict_all[mode][model_name]
            plt.plot(range(1, len(f1_list)+1), f1_list, marker='o', label=f'{model_name} ({mode})')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Model F1 Score Comparison (Sampling Methods)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.8,1])
    plt.savefig(os.path.join(out_dir, "model_f1_comparison_sampling.png"), bbox_inches='tight')
    plt.close()
    print(f"\nAll results and metrics saved in {out_dir}")
    print(f"Run time: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()
