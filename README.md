# Interview Proctoring Model

**Version:** 1.0.0

**Status:** Completed

**Release date:** 31 October 2026

An interview behaviour-monitoring project that extracts MediaPipe face features from recorded videos and live webcam footage, then uses trained temporal deep-learning models to classify:

1. **Talking / Listening**
2. **Looking down**
3. **Reading**

The project is an assistive review tool, not a surveillance or cheating-proof system. Predictions can be affected by lighting, camera quality, glasses, face occlusion, head movement, and personal gaze differences, and must receive human review.

## Current Project Status

The complete offline pipeline and live application are implemented:

- Frame-level feature extraction from recorded videos.
- Timestamp-based 7.5 FPS data standardization.
- Subject-held-out dataset preparation and temporal windowing.
- Three-class ANN, DNN, CNN, CNN+LSTM, and DNN+LSTM evaluation.
- Binary reading-detection training and evaluation workflows.
- MediaPipe live webcam feature extraction.
- Live three-class CNN+LSTM inference with a 38-frame rolling buffer.
- Streamlit webcam dashboard with prediction probabilities, history, durations, and configurable sustained-reading alerts.
- Separate five-second reading-detector logic and score smoothing.

The selected live model is the three-class **CNN+LSTM trained for 35 epochs and held out on SUBJECT_06**.

## Final Live Application

The Streamlit application is implemented in:

- `app/streamlit_app.py` — dashboard layout, controls, theme, webcam UI, probability bars, timeline, durations, and reading alerts.
- `app/video_processor.py` — MediaPipe extraction, 7.5 FPS sampling, rolling 38-frame buffer, valid-window checks, model loading, and inference.

### Dashboard functions

- Start and stop the browser webcam with the `streamlit-webrtc` START/STOP control.
- Display the live camera feed with a compact camera indicator for detected or missing faces.
- Refresh displayed metrics with the **Refresh metrics** button below the camera control.
- Display the current behaviour and confidence.
- Display probabilities for Talking / Listening, Looking down, and Reading.
- Show recent prediction history and estimated detected durations.
- Configure Reading alert confidence from `0.50` to `0.95`.
- Configure consecutive Reading predictions; the default is 2.
- Display a sustained Reading review alert when the configured conditions are met.
- Reset safely when camera processing is interrupted or restarted.
- Recover from missing faces without interpolating across long gaps.

The probability colours are: green for Talking / Listening, and red for Looking down and Reading. The dashboard uses a black and neutral theme.

## Live Inference Pipeline

```text
Browser webcam
    -> streamlit-webrtc frame
    -> MediaPipe Face Landmarks
    -> pitch, yaw, horizontal gaze ratio
    -> sample at 7.5 FPS
    -> maintain rolling 38-frame window
    -> validate finite feature coverage
    -> CNN+LSTM inference
    -> probability, prediction, history, and review alert
```

### Live input contract

| Setting | Value |
|---|---|
| Window size | 38 frames |
| Sampling rate | 7.5 FPS |
| Approximate duration | 5 seconds |
| Feature count | 3 |
| Feature order | Pitch, yaw, horizontal gaze ratio |
| Model input shape | `(1, 38, 3)` |
| Output classes | Talking, Looking down, Reading |
| Live checkpoint | `models/cnn_lstm_classifier_35epochs_SUBJECT_06.pt` |

Short missing observations are represented by NaN values. A window is rejected when it contains more than 10% missing values, and live inference currently rejects any incomplete finite sequence rather than interpolating across missing frames.

## Model Selection and Evaluation

The final live architecture is the three-class CNN+LSTM. It was compared on the SUBJECT_06 held-out fold with the flat DNN and DNN+LSTM temporal baseline.

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro ROC AUC | Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN+LSTM | 77.72% | 78.18% | 78.12% | 78.18% | 77.38% | 90.67% | 0.8542 |
| Flat DNN | 72.81% | - | - | - | 72.43% | - | 0.8069 |
| DNN+LSTM | 73.08% | - | - | - | 72.38% | - | 0.8099 |

CNN+LSTM achieved the highest reported accuracy and macro F1 among the temporal comparison models, with stronger Reading recall than the other 35-epoch models. The single held-out-subject result is preliminary and should be repeated with leave-one-subject-out validation and multiple random seeds before broad deployment.

The model implementation is in `src/model/cnn_lstm_model.py`. Training and evaluation scripts are available under `scripts/`, and detailed findings are recorded in `findings.md` and `context.md`.

## Launch Instructions

### Live Streamlit dashboard

Run from the project root with the configured Python 3.12 environment:

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

Allow camera permission in the browser. Use the single webcam START/STOP control, then wait for enough valid 38-frame windows before expecting a prediction.

### Offline feature extraction

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe scripts\run_extraction.py
```

### Offline training

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe scripts\run_training.py --model-type cnn_lstm --test-subject SUBJECT_06 --epochs 35
```

Supported training model types include `cnn`, `dnn`, `ann`, `cnn_lstm`, and `dnn_lstm`.

### Model evaluation

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe scripts\run_evaluation.py --help
```

### Five-second binary reading detector

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe scripts\run_reading_training.py --model-type cnn --test-subject SUBJECT_06 --epochs 35
```

## Project Structure

```text
app/
  streamlit_app.py       Streamlit dashboard and browser webcam UI
  video_processor.py     Live feature extraction and CNN+LSTM inference

configs/
  config.yaml            Dataset, windowing, feature, and alert settings

data/
  raw/                   Raw source videos, kept out of Git when large/private
  features*.csv          Extracted and timestamp-standardized frame features

models/
  cnn_lstm_classifier_35epochs_SUBJECT_06.pt  Selected live checkpoint
  *.pt                    Additional trained comparison checkpoints

scripts/
  run_extraction.py      Build frame-level feature CSV
  add_video_timestamps.py Standardize source observations to timestamps/FPS
  run_training.py        Train three-class models
  run_reading_training.py Train binary reading models
  run_evaluation.py      Evaluate saved checkpoints
  run_eda.py             Generate exploratory analysis and findings

src/
  capture/               Webcam and recorded-video frame generators
  features/              Feature extraction and motion/resampling helpers
  landmarks/             MediaPipe FaceMesh, head pose, and gaze calculations
  model/                 ANN, CNN, DNN, CNN+LSTM, DNN+LSTM, and detector models
  scoring/               Rolling prediction and reading-alert logic
  windowing/             Sequence construction for temporal models

results/                 Training and evaluation output
notebooks/               Exploration notebooks
findings.md              Data, model, and evaluation findings
context.md               Complete project history and release context
HOWTO.md                 Detailed offline and live usage guide
```

## Core Source Functions

### Live application

- `LiveInferenceProcessor.process_frame()` extracts and samples features, updates the rolling sequence, and runs inference.
- `LiveInferenceProcessor._predict_locked()` validates the current window and executes the CNN+LSTM model.
- `LiveInferenceProcessor.snapshot()` returns the latest prediction and buffer count safely across threads.
- `LiveInferenceProcessor.history_snapshot()` returns prediction history for the dashboard.
- `LiveInferenceProcessor.reset()` clears sequence and prediction state.
- `WebcamProcessor.recv()` converts browser frames, processes them, and returns the webcam frame to WebRTC.

### Feature extraction and landmarks

- `run_extraction.py` converts raw recordings into labelled frame-level features.
- `head_pose.get_head_pose()` calculates pitch and yaw.
- `gaze.get_gaze_ratio()` calculates horizontal and vertical eye gaze ratios.
- `face_mesh.get_landmarks()` processes a frame and returns facial landmarks when a face is detected.

### Training and models

- `build_model()` constructs the three-class CNN+LSTM architecture.
- `CNNLSTMClassifier.forward()` converts `(batch, frames, features)` into class probabilities.
- Training scripts save subject-held-out checkpoints to `models/` and print training/validation results.
- Evaluation scripts report accuracy, class metrics, confusion matrices, and supported probability metrics.

## Dependencies

Primary dependencies include:

- Python 3.12
- PyTorch
- OpenCV
- MediaPipe
- NumPy and Pandas
- Streamlit and streamlit-webrtc
- Scikit-learn
- Matplotlib and Seaborn
- PyAV

Install the project environment from the repository root:

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

## Tests

Run the focused unit tests:

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe -m unittest tests.test_gaze tests.test_landmarks tests.test_windowing -v
```

Also compile the application modules before release:

```powershell
D:\Interview-Proctoring-Model\.venv312\Scripts\python.exe -m compileall -q app src scripts
```

## Known Limitations

- The primary model result is based on one held-out subject and is not a definitive generalization benchmark.
- No calibrated live probability artifact has been added; confidence should not be interpreted as cheating probability.
- Talking and Reading remain confused in the reported single-fold evaluation.
- Glasses, glare, lighting, camera angle, occlusion, and partial face detection affect feature quality.
- Lack of interpolation across long gaps prevents false continuity, but may reduce live coverage.
- Reading and downward gaze can overlap behaviorally and are mutually exclusive labels in the evaluated three-class dataset.
- The output indicates visual behaviour patterns only. It does not establish intent, academic dishonesty, or misconduct.

## Recommended Future Work

1. Validate across all subjects and repeated seeds.
2. Calibrate class probabilities for live confidence thresholds.
3. Improve Talking/Reading separation with additional balanced data.
4. Separate a Reading review signal from a definitive misconduct decision.
5. Add recording consent, privacy controls, and automated data deletion.
6. Add deployment authentication and secure camera-session handling.
7. Add long-running telemetry for false-alert rate and detection delay where ethically and legally appropriate.

## License and Responsible Use

Obtain informed consent before recording or analysing interview candidates. Follow applicable employment, privacy, biometric-data, accessibility, and anti-discrimination requirements. Human decision-makers should review context and evidence rather than use model output as an automatic adverse-employment signal.