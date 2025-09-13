# Human Engagement LSTM Analysis - Setup & Usage Guide

## 1. Environment Setup

### Prerequisites

- **Python 3.13** (recommended: use [pyenv](https://github.com/pyenv/pyenv) or [conda](https://docs.conda.io/en/latest/) to manage Python versions)
- **pip** (comes with Python)
- **ffmpeg** (for audio extraction)
- **OpenCV** (for video processing)
- **PyTorch** (for LSTM models)
- **opensmile** (for audio features)
- **moviepy** (for video segmenting)
- **scikit-learn** (for metrics)
- **matplotlib** (for plotting)
- **h5py** (for annotation parsing)
- **pandas** (for CSV handling)
- **tqdm** (for progress bars)

### Install system dependencies

```sh
sudo apt-get update
sudo apt-get install ffmpeg libsmile python3-opencv
```

### Create and activate a virtual environment

```sh
python3.13 -m venv .venv
source .venv/bin/activate
```

### Install Python dependencies

```sh
pip install torch numpy pandas scikit-learn matplotlib tqdm h5py opensmile moviepy opencv-python
```

---

## 2. Data Preparation

### Organize your data folders as follows:

```
Lego_data/
    Train/
    Validation/
    Test/
Labelled_data_list.txt
```

- Each session folder should contain the required video files (`FC1_L.mp4`, `FC2_L.mp4`), transcripts (`*_lego.srt`), and annotation files (`annotations_raw.hdf5` or zipped).

---

## 3. Segment Videos and Prepare CSVs

### For Training Data

```sh
python Sampling_train.py
```

### For Validation and Testing Data

```sh
python Sampling_full_folder.py
```

---

## 4. Label Segments

```sh
python Labelling_automated.py
```

- Follow the prompts to label each segment as "low", "medium", or "high".

---

## 5. Feature Extraction

### Extract features (audio + visual, no stats):

```sh
python Feature_extraction_V2.0.py
```

- This will generate `.npy` feature files and metadata files for each session.

---

## 6. LSTM Training & Evaluation

### LSTM (no stats):

```sh
python LSTM_No_Stat.py
```

- Choose which models to run and which sampling method (normal, oversample, undersample).
- Results and plots are saved in the [`LSTM_results`](LSTM_results ) folder.

---

## 7. Additional Scripts

- **lib/** contains helper modules for annotation parsing, evaluation, and visualization which are required to run:
```
python Feature_extraction_V2.0.py
```
- You can use these for custom analysis or baseline generation.

---

## Notes

- All scripts assume you run them from the project root.
- If you encounter memory errors, try lowering batch size or running on CPU.
- For troubleshooting, check that all `.npy` and CSV files match the expected shapes and formats.

---

## Example Workflow

```sh
# 1. Segment videos and create CSVs
python Sampling_train.py
python Sampling_full_folder.py

# 2. Label segments
python Labelling_automated.py

# 3. Extract features
python Feature_extraction_V2.0.py

# 4. Analyze label distribution
python label manager.py

# 5. Train and evaluate LSTM models
python LSTM_No_Stat.py
```

---

## Contact

For issues or questions, please open an issue on GitHub.
