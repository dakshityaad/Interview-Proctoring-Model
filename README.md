# Interview Proctoring Model

**Version:** 1.0.0

**Status:** Complete

**Repository:** [dakshityaad/Interview-Proctoring-Model](https://github.com/dakshityaad/Interview-Proctoring-Model)

An end-to-end interview-behaviour analysis project that converts recorded interview videos into facial head-pose and gaze features, compares several PyTorch neural-network architectures, and deploys the best temporal model in a live Streamlit webcam dashboard.

The system predicts three mutually exclusive behaviours:

1. Talking / Listening
2. Looking Down
3. Reading

This project is an **assistive review tool**, not a cheating detector. Its predictions are probabilistic and can be affected by camera quality, lighting, glasses, posture, face visibility, individual differences, and dataset bias. Predictions must not be used as proof of misconduct or as an automatic employment decision.

---

## 1. Project Overview

The project follows one continuous machine-learning pipeline:

```text
Recorded interview video
        ↓
OpenCV frame extraction
        ↓
MediaPipe Face Mesh landmarks
        ↓
Head-pose pitch and yaw
        ↓
Horizontal and vertical eye gaze ratios
        ↓
Timestamp and 7.5 FPS standardization
        ↓
38-frame temporal windows
        ↓
ANN / DNN / CNN / CNN+LSTM / DNN+LSTM comparison
        ↓
Subject-held-out evaluation
        ↓
Selected CNN+LSTM checkpoint
        ↓
Live webcam + Streamlit inference
```

The version 1.0.0 application uses the same feature definitions and temporal window shape as offline training:

| Live inference setting | Value |
|---|---|
| Feature order | `pitch`, `yaw`, `gaze_ratio` |
| Sampling rate | 7.5 FPS |
| Temporal window | 38 frames |
| Approximate sequence duration | 5 seconds |
| Input tensor | `(batch, 38, 3)` |
| Output | Three class probabilities |
| Selected checkpoint | `models/cnn_lstm_classifier_35epochs_SUBJECT_06.pt` |
| Class mapping | Talking = 0, Looking down = 1, Reading = 2 |

---

## 2. Dataset Creation

### 2.1 Raw recordings

Six subjects were recorded performing three activities:

- Talking
- Looking down
- Reading

The repository's raw-video layout is:

```text
data/raw/
├── SUBJECT_01_TALKING.mp4
├── SUBJECT_01_LOOKING.mp4
├── SUBJECT_01_READING.mp4
├── SUBJECT_02_TALKING.mp4
├── SUBJECT_02_LOOKING.mp4
├── SUBJECT_02_READING.mp4
...
└── SUBJECT_06_READING.mp4
```

The project also supports a nested layout:

```text
data/raw/<subject>/<activity-video>
```

Activity names map to numeric labels in the extraction code:

| Activity | Label |
|---|---:|
| Talking | 0 |
| Looking down | 1 |
| Reading | 2 |

This naming convention makes subject and class boundaries explicit, which is important because random frame-level splitting would allow near-identical windows from the same subject and recording to appear in both training and validation data.

### 2.2 Frame and landmark extraction

`scripts/run_extraction.py` opens every supported source video and passes its frames to the OpenCV and MediaPipe pipeline implemented under `src/landmarks/`.

For each frame:

1. OpenCV decodes the frame.
2. MediaPipe Face Mesh returns normalized facial landmarks.
3. Six landmarks are used to estimate head pose with OpenCV `solvePnP`.
4. Eye-corner and iris landmarks are used to calculate gaze ratios.
5. The original frame number and source FPS are written with the measurements.
6. Frames without a reliable face or pose are skipped rather than filled with invented values.

The resulting frame-level CSV has this schema:

```text
subject_id,
class_label,
frame_number,
timestamp_seconds,
source_fps,
pitch,
yaw,
gaze_ratio
```

### 2.3 Head-pose features

The pitch and yaw calculation is in `src/landmarks/head_pose.py`.

MediaPipe landmark points are transformed from normalized image coordinates into pixels. The OpenCV pose-estimation routine fits an approximate three-dimensional generic face model to these six landmarks:

- Nose tip
- Chin
- Left eye outer corner
- Right eye outer corner
- Left mouth corner
- Right mouth corner

The returned head-pose values use degrees:

- **Pitch:** positive means the head is tilted upward; negative means downward.
- **Yaw:** positive means rotation toward the subject's left; negative means toward the subject's right in the documented convention.

### 2.4 Gaze feature

The gaze calculation is in `src/landmarks/gaze.py`.

For each eye, the code normalizes the iris position horizontally and vertically between eye landmarks. Both eyes are averaged to create one horizontal gaze ratio and one vertical gaze ratio. The final models currently use the horizontal gaze ratio (`gaze_ratio`), while pitch captures the dominant vertical head movement.

The ratio calculation is scale-invariant: dividing iris movement by eye width or height limits the effect of face-camera distance.

### 2.5 Dataset statistics

The refreshed frame-level dataset contains **77,519 usable frames**, six subjects, and all three activities for every subject.

| Activity | Frame count | Share |
|---|---:|---:|
| Talking | 26,462 | 34.14% |
| Looking down | 20,970 | 27.05% |
| Reading | 30,087 | 38.81% |

| Subject | Talking | Looking down | Reading | Total |
|---|---:|---:|---:|---:|
| SUBJECT_01 | 5,060 | 4,218 | 10,385 | 19,663 |
| SUBJECT_02 | 2,250 | 2,191 | 2,791 | 7,232 |
| SUBJECT_03 | 9,339 | 6,471 | 5,765 | 21,575 |
| SUBJECT_04 | 2,508 | 1,054 | 2,388 | 5,950 |
| SUBJECT_05 | 2,272 | 1,029 | 2,401 | 5,702 |
| SUBJECT_06 | 5,033 | 6,007 | 6,357 | 17,397 |

The recordings were not all captured at the same FPS. Observed source rates include approximately 7.5, 16.7, and 30 FPS. Raw frame count therefore is not comparable with duration.

### 2.6 Timestamp standardization

`scripts/add_video_timestamps.py` reads each source video's FPS and calculates:

```text
timestamp_seconds = original_frame_number / source_fps
```

During window construction, `src/windowing/make_windows.py` resamples each subject/activity sequence to a common **7.5 FPS timeline**. The common sampling rate connects offline training to live inference, where the browser processor also contributes approximately 7.5 samples per second.

A final three-class window therefore represents approximately five seconds of observations:

```text
38 frames ÷ 7.5 frames/second ≈ 5 seconds
```

### 2.7 Temporal windowing and missing data

Features are grouped by subject and class. Each recording sequence is divided into overlapping windows:

- Window size: 38 frames
- Stride: 10 frames
- Maximum allowed missing values: 10% of a window

Short internal gaps can be filled by linear interpolation, limited to a maximum gap duration. Long gaps and boundary gaps remain non-finite so they cannot be interpreted as real observations. Windows that retain excessive missing data are rejected.

Live inference uses the same 38-frame shape but does not interpolate across missing frames. A live window is rejected when it is not finite, preventing a silent prediction from a fabricated sequence.

---

## 3. Train and Validation Strategy

The main temporal models use **subject-held-out validation**. For the reported comparison, `SUBJECT_06` is held out while the remaining subjects train the model:

```text
Training: SUBJECT_01–SUBJECT_05
Validation: SUBJECT_06
```

This tests whether the model recognizes behaviour patterns from a person not seen during training. It is more suitable than random frame splitting because frames from one recording can be highly correlated.

The subject-held-out evaluation is still one experimental fold. Repeated seeds and leave-one-subject-out validation across every subject are needed before the result can be described as robust.

### Cross-validation limitation

Only temporal comparisons are shown against one fixed held-out subject in the current reported results. Repeating the experiment with all subjects as held-out folds would provide confidence intervals and reveal whether performance depends heavily on one person or camera setup.

---

## 4. Architecture Comparison

Five architecture families were built in `src/model/`.

### ANN

A dense neural network evaluates the relationship between flattened frame/window values and activity classes. It is the simplest point-by-point comparison model.

### DNN

The three-class DNN is a deeper multilayer feed-forward network that receives flattened temporal windows. It captures nonlinear relationships between features but does not contain an explicit recurrent sequence layer.

### CNN

The CNN applies local one-dimensional convolutions across the temporal dimension. It learns short local patterns such as pitch transitions, gaze changes, and their combinations.

### CNN+LSTM

The selected architecture is implemented in `src/model/cnn_lstm_model.py`:

```text
Input: (batch, 38 frames, 3 features)
        ↓
Conv1d 3 → 32 channels, kernel 5
        ↓
ReLU
        ↓
Conv1d 32 → 32 channels, kernel 3
        ↓
ReLU
        ↓
LSTM, 64 hidden units
        ↓
Dropout
        ↓
Linear 64 → 3 classes
        ↓
Softmax probabilities
```

The convolutional stages learn local temporal motifs. The LSTM processes their evolution across the complete window and retains longer-range sequence context.

### DNN+LSTM

The DNN+LSTM is implemented in `src/model/dnn_lstm_model.py`. Every frame is projected through dense layers before the sequence is processed by a 64-unit LSTM:

```text
Input (38, 3)
  → frame-wise Dense/ReLU projection
  → 64-unit LSTM
  → Dense classifier
  → three probabilities
```

It provides a useful comparison between convolutional temporal feature extraction and frame-wise dense temporal projection.

---

## 5. 35-Epoch Model Results

All training details below correspond to the documented 35-epoch subject-held-out experiments evaluated on `SUBJECT_06`, unless another model/table states otherwise.

### 5.1 Flat three-class models

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | ROC AUC | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 68.30% | 68.59% | 67.92% | 68.59% | 66.97% | 85.11% | 1.4977 |
| ANN | 70.69% | 71.35% | 72.19% | 71.35% | 70.53% | 85.60% | 1.1929 |
| DNN | 72.81% | 74.27% | 75.91% | 74.27% | 72.43% | 89.18% | 0.8069 |

Per-class F1 for those three models:

| Model | Talking | Looking down | Reading |
|---|---:|---:|---:|
| CNN | 55.07% | 91.56% | 54.27% |
| ANN | 59.10% | 94.70% | 57.79% |
| DNN | 64.10% | 95.11% | 58.09% |

The DNN was the strongest architecture in the original flat CNN/ANN/DNN comparison. Looking Down is separated especially well by head pitch, while Talking and Reading remain harder to distinguish from the three available signals.

### 5.2 Temporal and flat comparison

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC AUC | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DNN | 72.81% | 74.27% | 75.91% | 74.27% | 72.43% | 72.66% | 89.18% | 0.8069 |
| CNN+LSTM | 77.72% | 78.18% | 78.12% | 78.18% | 77.38% | 77.87% | 90.67% | 0.8542 |
| DNN+LSTM | 73.08% | 73.33% | 73.23% | 73.33% | 72.38% | 73.01% | 89.78% | 0.8099 |

### 5.3 CNN+LSTM result

Compared with the flat DNN, the selected CNN+LSTM improved:

- Accuracy by **4.91 percentage points**
- Balanced accuracy by **3.91 percentage points**
- Macro F1 by **4.95 percentage points**
- Macro ROC AUC by **1.49 percentage points**

It also increased Reading recall from 46.29% to 60.07% and Reading F1 from 58.09% to 67.19%.

CNN+LSTM confusion matrix, with rows representing actual classes and columns representing predicted classes:

| Actual \ Predicted | Talking | Looking down | Reading |
|---|---:|---:|---:|
| Talking | 162 | 2 | 44 |
| Looking down | 0 | 254 | 9 |
| Reading | 113 | 0 | 170 |

The most important remaining errors are Reading windows predicted as Talking and Talking windows predicted as Reading. The low probability of confusion with Looking Down shows that pitch is still the strongest available Looking Down signal.

CNN+LSTM's log loss is slightly worse than the flat DNN. Therefore its raw softmax probability should not be presented as a calibrated likelihood until reliability calibration and larger subject coverage are available.

### 5.4 DNN+LSTM result

Compared with the flat DNN, DNN+LSTM:

- Improved accuracy by **0.27 percentage points**
- Improved Reading recall by approximately **7.42 percentage points**
- Reduced macro F1 by approximately **0.05 percentage points**
- Reduced macro precision by **2.68 percentage points**

DNN+LSTM confusion matrix:

| Actual \ Predicted | Talking | Looking down | Reading |
|---|---:|---:|---:|
| Talking | 145 | 13 | 50 |
| Looking down | 3 | 254 | 6 |
| Reading | 128 | 3 | 152 |

Its per-class F1 scores were approximately:

| Talking | Looking down | Reading |
|---:|---:|---:|
| 59.92% | 95.31% | 61.91% |

CNN+LSTM remained the strongest temporal three-class classifier in the reported comparison.

---

## 6. Binary Reading Detector Comparison

The project also includes an additional binary Reading detector trained with six motion features derived from the base three-feature sequence:

- Pitch velocity and acceleration
- Yaw velocity and acceleration
- Gaze velocity and acceleration
- Horizontal motion

The binary dataset uses a five-second 38-frame window at 7.5 FPS and converts every activity to Reading versus Not Reading.

| Model | Accuracy | Balanced accuracy | Reading precision | Reading recall | Reading F1 | Macro F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 53.69% | 46.14% | 28.79% | 15.86% | 20.46% | 43.90% | 65.64% |
| ANN | 62.25% | 56.76% | 49.60% | 34.70% | 40.83% | 56.56% | 72.34% |
| DNN | 58.16% | 55.26% | 44.19% | 43.63% | 43.91% | 55.27% | 66.44% |

The ANN achieved the best binary accuracy and ROC AUC, while the DNN had the best Reading recall and F1. However, all binary detectors need substantial improvement before operational use. This binary experiment is separate from the selected three-class CNN+LSTM live system; the application uses the three-class model for predictions and applies temporal consecutive-window logic to Reading review alerts.

---

## 7. Live Webcam Application

The live application is implemented in:

- `app/streamlit_app.py`: Streamlit dashboard, browser WebRTC controls, history, durations, probabilities, alerts, and styling.
- `app/video_processor.py`: frame sampling, feature extraction, model loading, rolling sequence buffer, validation, and prediction logic.

### WebRTC data flow

```text
Browser webcam frame
        ↓
streamlit-webrtc media processor
        ↓
BGR OpenCV frame
        ↓
MediaPipe Face Mesh
        ↓
Pitch, yaw, horizontal gaze ratio
        ↓
7.5 FPS sampler
        ↓
38-frame rolling NumPy sequence
        ↓
CNN+LSTM checkpoint
        ↓
Three-class Softmax output
        ↓
Dashboard probabilities and history
```

### Dashboard functions

- One integrated WebRTC start/stop camera control.
- Live camera feed and face-detection status.
- “Talking / Listening,” “Looking Down,” and “Reading” labels.
- Confidence and all three probabilities.
- Rolling prediction history and behavior durations.
- Refresh metrics control.
- Configurable Reading review threshold and consecutive-window count.
- Dark black-neutral interface designed without blue UI accents.
- Safe handling of missing faces and interrupted cameras.
- No interpolation across long live gaps.

The sustained Reading alert defaults to two consecutive predictions at confidence 0.75. It intentionally uses consecutive evidence rather than treating a single noisy class prediction as definite behavior:

```text
Potential reading behaviour detected — review required.
```

This message is a review prompt, not proof of cheating.

---

## 8. Repository Structure

```text
app/
├── streamlit_app.py       Live Streamlit dashboard
└── video_processor.py     Live extraction, buffering, and inference

configs/
└── config.yaml            Paths, classes, windowing, and alert defaults

data/
├── raw/                   Recorded source videos
├── features.csv           Extracted frame-level feature data
├── features_six_subjects.csv
└── features_7_5fps.csv    Timestamp-standardized feature data

models/
├── cnn_lstm_classifier_35epochs_SUBJECT_06.pt
└── other comparison checkpoints and training artifacts

notebooks/
└── dataset and model exploration notebooks

scripts/
├── run_extraction.py
├── add_video_timestamps.py
├── run_eda.py
├── run_training.py
├── run_evaluation.py
└── run_reading_training.py

src/
├── capture/               Webcam and recorded-video readers
├── features/              Dataset preparation and motion features
├── landmarks/             MediaPipe, head pose, and gaze extraction
├── model/                 ANN, DNN, CNN, LSTM, and binary detectors
├── scoring/               Rolling and binary reading scoring
└── windowing/             Temporal sequence and resampling logic

tests/
├── test_gaze.py
├── test_landmarks.py
└── test_windowing.py

results/                   Evaluation files and generated plots
findings.md                Dataset and experiment findings
context.md                 Project history and release context
HOWTO.md                   Detailed operational instructions
README.md                  Project overview and user guide
```

---

## 9. Installation

### Requirements

The project was developed with Python 3.12 and Windows. The intended interpreter is:

```text
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe
```

Key packages include:

- PyTorch
- OpenCV
- MediaPipe 0.10.35
- NumPy
- Pandas
- SciPy
- PyYAML
- scikit-learn
- Matplotlib and Seaborn
- Streamlit
- streamlit-webrtc
- PyAV

### Create or activate an environment

```powershell
cd D:\Interview-Proctoring-Model
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If browser execution policy blocks activation, activate through Command Prompt:

```bat
.venv312\Scripts\activate.bat
```

Alternatively, run the virtual environment's Python executable directly as shown in the commands below.

---

## 10. Running the Project

### Extract features from raw videos

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe scripts\run_extraction.py
```

### Add timestamps and source FPS

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe scripts\add_video_timestamps.py
```

### Run exploratory data analysis

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe scripts\run_eda.py
```

### Train a three-class model

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe scripts\run_training.py --model-type cnn_lstm --test-subject SUBJECT_06 --epochs 35
```

Supported model types:

```text
cnn
ann
dnn
cnn_lstm
dnn_lstm
```

Additional useful options include:

```powershell
--features-csv data/features.csv
--model-path models/model.pt
--test-subject SUBJECT_06
--epochs 35
--batch-size 64
--learning-rate 0.001
```

### Train the binary Reading detector

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe scripts\run_reading_training.py --model-type dnn --test-subject SUBJECT_06 --epochs 35
```

### Launch the live dashboard

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

Allow browser camera access, start the WebRTC camera control, and wait until approximately 38 valid samples have accumulated before expecting a prediction.

---

## 11. Important Source Functions

### Feature extraction

- `extract_features_from_video()` in `src/features/extract_features.py` processes one recording.
- `build_features_csv()` converts all recordings into a labeled dataset.
- `get_head_pose()` in `src/landmarks/head_pose.py` estimates pitch and yaw.
- `get_gaze_ratio()` in `src/landmarks/gaze.py` calculates normalized eye-gaze ratios.

### Windowing

- `resample_features()` standardizes sequences to a common FPS.
- `fill_short_missing_gaps()` interpolates short internal gaps during offline preparation.
- `slice_into_windows()` creates overlapping fixed-length sequences.
- `build_dataset()` joins windows with labels and subject IDs.

### Models

- `build_model()` is implemented for ANN, DNN, CNN, CNN+LSTM, and DNN+LSTM architectures.
- `CNNLSTMClassifier.forward()` accepts `(batch, frames, features)` and returns softmax probabilities.
- `train_model()` trains PyTorch models.
- `save_model()` writes checkpoints for later offline and live inference.

### Live inference

- `LiveInferenceProcessor.process_frame()` handles one browser frame.
- `LiveInferenceProcessor._predict_locked()` validates and predicts a sequence.
- `LiveInferenceProcessor.snapshot()` returns the latest prediction and sequence state.
- `LiveInferenceProcessor.history_snapshot()` supplies prediction history to Streamlit.
- `LiveInferenceProcessor.reset()` clears rolling state.
- `WebcamProcessor.recv()` connects the browser frame processor to the stream.

---

## 12. Testing and Validation

Run the unit tests from the repository root:

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe -m unittest tests.test_gaze tests.test_landmarks tests.test_windowing -v
```

Compile the Python source and application modules:

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe -m compileall -q app src scripts
```

The validation checklist should include:

- Raw recordings resolve to subject and activity labels correctly.
- MediaPipe returns finite pitch, yaw, and gaze features.
- Resampling produces the expected 7.5 FPS timeline.
- Every temporal window has shape `(38, 3)`.
- Subject-held-out data contains no validation subject records from training.
- The selected checkpoint loads and produces `(1, 3)` probabilities.
- Live camera interruption leaves no stale prediction sequence.
- Face detection recovery works after temporary occlusion.
- Unknown camera failures are reported instead of crashing the dashboard.
- A sustained Reading signal requires configurable consecutive predictions.

---

## 13. Known Limitations

- The selected result comes from one held-out subject; it is not proof of population-level performance.
- No calibration curve or held-out probability calibration has yet been produced.
- Talking and Reading overlap strongly in head pose and gaze.
- Looking Down is strongly represented by negative pitch but may still appear in other contexts.
- Glasses, glare, eye shape, facial hair, distance, lighting, occlusion, and camera angle affect features.
- Eye gaze alone cannot establish reading a specific document or screen.
- The live browser machine, frame timing, and server load can change the effective sampling rate.
- Consecutive Reading windows reduce noise but do not eliminate false alerts.
- The binary Reading detector should not be treated as production-ready.
- The model does not establish intent, honesty, engagement quality, or cheating.

---

## 14. Recommended Improvements

1. Run leave-one-subject-out validation and report means and confidence intervals.
2. Repeat training with multiple seeds and document variance.
3. Balance subject, class, camera, and lighting diversity.
4. Calibrate class probabilities with temperature scaling or a held-out validation set.
5. Add false-alert, missed-alert, and prediction-delay metrics.
6. Add optional calibration videos for each candidate or camera environment.
7. Improve Reading discrimination using longer temporal windows or reliable screen/document cues.
8. Separate “Talking / Listening” into different labels only if the data and intended behavior are well defined.
9. Add explicit privacy, consent, data-retention, encryption, and deletion controls.
10. Test accessibility and ensure any review workflow allows human context and appeal.

---

## 15. Responsible Use

Interview monitoring can affect candidate privacy and employment outcomes. Before using this project in a real process:

- Obtain informed consent.
- Explain what is measured and why.
- Do not store camera frames unless necessary and explicitly authorized.
- Follow biometric-data, workplace, accessibility, anti-discrimination, and privacy laws.
- Keep human reviewers independent of automatic model output.
- Allow candidates to explain environmental conditions or disabilities affecting camera signals.
- Never use a Reading alert as proof of misconduct.
- Prefer aggregate model evaluation and workflow monitoring over individual automated decisions.

---

## 16. Project Documentation

Additional project records:

- [context.md](context.md): complete implementation history, decisions, version records, and release details.
- [findings.md](findings.md): dataset analysis and model-comparison results.
- [HOWTO.md](HOWTO.md): setup and detailed command reference.
- [models/cnn.md](models/cnn.md): CNN documentation.

---

## 17. Version

**Version 1.0.0 — Project completed and released 31 October 2026.**

This release includes dataset preparation, feature extraction, timestamp standardization, architecture comparison, subject-held-out evaluation, CNN+LSTM model selection, the binary Reading experiment, and a live Streamlit webcam dashboard.
