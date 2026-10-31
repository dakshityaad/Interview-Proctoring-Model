# Project Context

## Version 0.0.0

**Note:** The project structure and initial documentation were created from the planned design.

**Project:** Interview Screen-Reading Detector
**Purpose:** Detect normal behavior, looking down, and reading during online interviews using webcam gaze and head-pose signals.

### Current Files

- `README.md` - Project problem, theory, data plan, limitations, and model phases.
- `build_sheet.md` - Planned source structure, file responsibilities, functions, and build order.
- `context.md` - This project-only summary and action log.
- `requirements.txt` - Required Python dependencies from the README.
- Project scaffold - Empty placeholders created for configs, data, source packages,
	scripts, models, results, tests, notebooks, and the application.

### Action Log

- Version 0.0.0: Added `context.md` with the current project structure and file summary.
- Version 0.0.0: Renamed `Build_Sheet (1).md` to `build_sheet.md`.
- Version 0.0.0: Created the README-based project directory and file scaffold.
- Version 0.0.0: Left scaffold files empty and populated only `requirements.txt` with
	the listed modeling, data, visualization, and Streamlit dependencies.

## Version 0.0.1

**Note:** The empty project scaffold and dependency list were completed and checked.

### Current Files

- `README.md` - Project requirements and modeling plan.
- `build_sheet.md` - Planned file structure and implementation order.
- `context.md` - Project history and version notes.
- `requirements.txt` - Initial Python dependency list.
- Project scaffold - Empty source, test, app, script, model, data, and notebook files.

### Action Log

- Created the empty project files and folders based on the structure in `README.md`.
- Kept all source, test, configuration, model, data, notebook, and app files empty.
- Added the required package list to `requirements.txt`.
- Confirmed that the scaffold files contain no code.

## Version 0.0.2

**Note:** The first working webcam face-landmark prototype was implemented and published.

### Current Files

- `src/landmarks/face_mesh.py` - OpenCV and MediaPipe webcam face-mesh prototype.
- `models/face_landmarker.task` - MediaPipe face-landmark model asset.
- `requirements.txt` - Webcam and MediaPipe dependencies.
- `README.md`, `build_sheet.md`, and `context.md` - Project documentation and history.
- Remaining project directories - Scaffolded files for future features.

### Action Log

- Implemented a working face-landmark prototype in `src/landmarks/face_mesh.py` using OpenCV and MediaPipe.
- Added a live webcam loop that reads frames, processes the face mesh, draws landmarks, and exits on `q`.
- Updated `requirements.txt` to include the needed webcam/media tooling for live face tracking.
- Added the MediaPipe face landmark model asset at `models/face_landmarker.task`.
- Verified the repo state and committed the recent changes with a message: `Update interview proctoring logic`.
- Pushed the latest version to the remote `main` branch on GitHub.

## Version 0.0.3

**Note:** MediaPipe compatibility and video timestamp problems were fixed for the Python 3.12 setup.

### Current Files

- `src/landmarks/face_mesh.py` - Supports legacy MediaPipe and Tasks APIs with video timestamps.
- `models/face_landmarker.task` - MediaPipe Tasks model asset.
- `requirements.txt` - Pins `mediapipe==0.10.35`.
- `.venv312/` - Python 3.12 project environment.
- `context.md` - Compatibility fixes and recommended run command.

### Action Log

- Diagnosed the MediaPipe runtime failure as an API and interpreter mismatch. The active
	`.venv` uses Python 3.14, while the project is configured to run with Python 3.12.
- Pinned `mediapipe==0.10.35` in `requirements.txt` and created the `.venv312` environment
	with Python 3.12.10.
- Updated `src/landmarks/face_mesh.py` to support the installed MediaPipe Tasks API by using
	`BaseOptions`, `FaceLandmarker`, `FaceLandmarkerOptions`, and `RunningMode` when the legacy
	`mp.solutions.face_mesh` API is unavailable.
- Fixed the video inference crash caused by passing timestamp `0` for every frame. The script
	now passes a strictly increasing millisecond timestamp to `detect_for_video`.
- Verified the updated script with `py_compile` and confirmed MediaPipe 0.10.35 imports and
	the Tasks API symbols successfully in `.venv312`.
- The MediaPipe startup warnings about XNNPACK and feedback tensors are informational and do
	not indicate a failure.
- The recommended command is:
	`.\.venv312\Scripts\python.exe src\landmarks\face_mesh.py`

## Version 0.0.4

**Note:** Gaze and head-pose overlays were added, and webcam tracking responsiveness was improved.

### Current Files

- `src/landmarks/face_mesh.py` - Face mesh, gaze overlay, head-pose overlay, and webcam demo.
- `src/landmarks/head_pose.py` - Head-pose estimation helper.
- `models/face_landmarker.task` - MediaPipe face-landmark model asset.
- `requirements.txt` - MediaPipe, OpenCV, and project dependencies.
- `context.md` - Version history and validation notes.

### Action Log

- Added iris-marker and gaze-position overlays plus head-pose axes to
	`src/landmarks/face_mesh.py`.
- Reduced visible eye-tracking latency by limiting the webcam capture buffer to one frame.
- Corrected the MediaPipe Tasks VIDEO-mode timestamp to use elapsed milliseconds instead of
	a 1 ms-per-loop counter, preserving strictly increasing real-time timestamps for tracking.
- Verified the updated face-mesh script with `py_compile`.

## Version 0.0.5

**Note:** Gaze, head-pose, and frame-capture logic were separated into modules, documented, and tested.

### Current Files

- `src/capture/webcam_stream.py` - Lazy webcam and recorded-video frame generators.
- `src/landmarks/face_mesh.py` - MediaPipe processing loop and visualization.
- `src/landmarks/gaze.py` - Scale-invariant `get_gaze_ratio()` implementation.
- `src/landmarks/head_pose.py` - Head-pose estimation implementation.
- `tests/test_gaze.py` - Gaze scale-invariance regression test.
- `models/face_landmarker.task` and `requirements.txt` - Runtime model and dependencies.
- `context.md` - Project history, current files, and action logs.

### Action Log

- Moved iris-center and eye-corner gaze-ratio calculation into
	`src/landmarks/gaze.py` as `get_gaze_ratio(landmarks, frame_width, frame_height)`.
- Kept head-pose estimation in `src/landmarks/head_pose.py` and updated the face-mesh
	overlay to call the separate gaze and head-pose modules.
- Added `tests/test_gaze.py` to verify that gaze ratios remain unchanged when the projected
	face scale is reduced, representing a face farther from the camera.
- Added explanatory comments to `src/landmarks/face_mesh.py` and `src/landmarks/gaze.py`
	covering the MediaPipe pipeline, backend selection, normalized coordinates, and scale
	invariance.
- Verified `face_mesh.py` and `gaze.py` with `py_compile`; the gaze regression test passed
	with `unittest`; editor diagnostics reported no errors.
- Started the live webcam demo successfully and confirmed MediaPipe initialized. Exact
	pitch/yaw/gaze before-and-after numeric comparison was not recorded.
- Added `src/capture/webcam_stream.py` with lazy `read_webcam_frames()` and
	`read_video_frames(path)` generators so frames are processed one at a time.
- Updated `src/landmarks/face_mesh.py` to use `read_webcam_frames()` instead of opening
	`cv2.VideoCapture` directly.
- Preserved the one-frame webcam capture buffer and verified the recorded-video generator
	with a temporary three-frame video.
- Re-verified the capture changes with `py_compile`, the existing test suite, diagnostics,
	and `git diff --check`.

## Version 0.0.6

**Note:** Configuration, feature extraction, and PyTorch dataset preparation
were implemented through creation of the combined feature CSV.

### Current Files

- `configs/config.yaml` - Windowing, alert, path, class, extraction, and feature schema settings.
- `src/features/extract_features.py` - Converts recorded videos into labeled frame-level features.
- `src/model/dataset.py` - Subject-level splitting and PyTorch tensor conversion.
- `scripts/run_extraction.py` - Thin extraction CLI.
- `data/features.csv` - Combined CSV output; row counts and frame-gap checks remain to be verified.

### Action Log

- Added configuration for window size `60`, stride `10`, alert defaults, paths, class mappings, video extensions, no-face policy, and feature columns.
- Added `PyYAML` to `requirements.txt` and verified that the configuration parses.
- Implemented video feature extraction using face landmarks, head pose, and horizontal gaze ratio.
- Chose to skip frames without a face or valid pose while preserving original frame numbers.
- Implemented combined CSV generation for nested subject folders and flat names such as `SUBJECT_01_READING.mp4`.
- Implemented PyTorch subject-level splitting and conversion to `float32` features and `int64` labels.
- Added explanatory comments and docstrings following the style of `src/landmarks/head_pose.py`.
- Verified imports, CLI help, compilation, diagnostics, whitespace, and tests: `1 passed`.
- The extraction CLI completed with exit code 0 and normal MediaPipe startup messages. CSV population, class counts, and frame-gap sanity checks are the next verification step.

## Version 0.1.0

**Note:** The first end-to-end CNN training pipeline was implemented and
validated on the extracted feature data.

### Current Files

- `src/windowing/make_windows.py` - Loads the feature CSV and creates labeled,
	 subject/class-safe temporal windows.
- `src/model/cnn_model.py` - Defines the 1D CNN classifier for temporal face
	 features.
- `src/model/train.py` - Provides the training loop, validation tracking, and
	 model checkpoint saving.
- `scripts/run_training.py` - CLI entry point for subject-level CNN training.
- `models/cnn_classifier.pt` - Trained CNN weights for the default training run.
- `results/cnn/cnn_classifier_subject02.pt` - CNN weights from the
	 `SUBJECT_02` held-out validation run.

### Action Log

- Implemented temporal window loading and slicing using the configured window
	 size of `60` frames and stride of `10` frames.
- Ensured windows do not cross subject/class recording boundaries.
- Implemented a 1D CNN with three convolutional layers, ReLU activations,
	 max-pooling, adaptive average pooling, dense layers, and three-class softmax
	 output.
- Implemented the forward/backward training loop with Adam optimization,
	 validation loss, and validation accuracy tracking.
- Implemented checkpoint saving to `models/cnn_classifier.pt`.
- Added CLI options for feature CSV path, model path, held-out subject, epochs,
	 batch size, and learning rate.
- Trained the CNN end to end using the real feature CSV. With `SUBJECT_03`
	 held out, training loss decreased from `0.7084` to `0.1830` and validation
	 accuracy reached `65.30%`.
- Evaluated a separate `SUBJECT_02` held-out fold. Training loss decreased from
	 `0.5943` to `0.2240` and validation accuracy reached `55.08%`.
- Confirmed the new Python modules compile successfully and the existing gaze
	 regression test passes.
- Confirmed the `.pt` checkpoint is a valid binary PyTorch file; it begins with
	 `PK` because PyTorch stores checkpoints in a ZIP-based format.

## Version 0.2.0

**Note:** Exploratory data analysis was completed and the project was extended
with a five-second binary reading-motion detector.

### EDA Findings

- `data/features.csv` contains `38,085` frame-level rows from
	`SUBJECT_01`, `SUBJECT_02`, and `SUBJECT_03`.
- The frame-level class counts are class 0: `16,649`, class 1: `12,880`,
	and class 2: `8,556`; class 2 is the minority class.
- No missing values were found in the extracted feature CSV.
- Class 1 has strongly negative average pitch (`-37.1563`), supporting the
	looking-down interpretation.
- Class 2 has a different yaw pattern and slightly higher average gaze ratio,
	which supports using temporal movement rather than posture alone.
- The existing 60-frame, stride-10 configuration produced `3,766` windows.
- SUBJECT_01 has no class 2 recording, so subject-level validation and
	per-class metrics remain important generalization safeguards.
- Full EDA details were recorded in `findings.md` and the analysis notebook was
	updated in `notebooks/exploration.ipynb`.

### Reading Detector Implementation

- Added `src/features/motion.py` to calculate velocity, acceleration, and
	horizontal-motion features from pitch, yaw, and gaze ratio.
- Extended `src/windowing/make_windows.py` with motion-aware binary reading
	windows while preserving the original three-class API.
- Added a default five-second window configuration in `configs/config.yaml`:
	`15 FPS`, `75 frames`, and a `0.5 second` stride.
- The new reading input contains 10 channels: the 3 original features plus 7
	temporal motion features.
- Added `scripts/run_reading_training.py` for class-weighted binary CNN
	training. The target is `0 = not reading` and `1 = reading`, with original
	classes 0 and 1 combined into not-reading.
- Added `src/scoring/reading_score.py` with hysteresis and consecutive-window
	requirements. The default alert starts at probability `0.70` for two
	consecutive windows and clears at probability `0.40` for two windows.
- Added `src/model/reading_detector.py`, a stateful streaming detector that
	accumulates 75 frames and returns `reading_probability` and `is_reading`.
- Added regression tests covering motion feature shape, five-second windows,
	and persistent alert behavior.

### Validation

- The real feature CSV produced `4,690` binary reading windows with shape
	`(75, 10)`.
- Focused project tests passed: `4 tests`, `OK`.
- The streaming detector returned no result before 75 frames and produced a
	prediction after the buffer filled.
- A one-epoch end-to-end training smoke test completed successfully and saved
	a checkpoint; the temporary smoke checkpoint was removed afterward.
- Pylance diagnostics and `git diff --check` reported no errors for the
	changed implementation files.

### Current Limitation / Next Integration Step

- The detector currently assumes the extracted feature stream is sampled at
	15 FPS. The raw video FPS is not persisted in `data/features.csv`.
- `app/video_processor.py` and `app/streamlit_app.py` are still empty. The next
	implementation step is to connect `ReadingDetector.update(pitch, yaw,
	gaze_ratio)` to the live MediaPipe frame-processing loop.

## Version 0.3.0 — ANN and DNN Improvements

**Note:** This version updates the research direction for improving the ANN and
DNN reading detectors while retaining the existing CNN as the primary temporal
baseline.

### Current findings

- CNN average subject-held-out accuracy: approximately 76.34%.
- ANN average subject-held-out accuracy: approximately 68.03%.
- DNN average subject-held-out accuracy: approximately 67.01%.
- All accuracy comparisons use 75-frame windows, representing 5 seconds at the
	configured 15 FPS.
- Subject-held-out evaluation is used, but CNN, ANN, and DNN comparisons should
	be made more rigorous with identical seeds and repeated folds.

### Recommended improvements

1. Normalize all 10 features using means and standard deviations computed only
	from training subjects to prevent held-out-subject leakage.
2. Reduce ANN/DNN input dimensionality by testing statistical window summaries
	such as mean, standard deviation, minimum, maximum, median, endpoints, slope,
	and energy.
3. Replace plain Adam training with AdamW, weight decay, learning-rate
	scheduling, and early stopping where appropriate.
4. Tune flattened standardized-window models before testing summarized-feature
	models.
5. Test a summarized-feature DNN such as `256 → 128 → 64` with BatchNorm,
	ReLU/GELU, and moderate dropout.
6. Add precision, recall, F1, confusion matrices, false alerts per minute,
	missed-reading duration, and detection-delay evaluation.
7. Report the mean and standard deviation across repeated random seeds for each
	subject-held-out fold.

### Architecture implications

- ANN and DNN currently flatten each `(75, 10)` window into 750 inputs.
- Flattening loses explicit temporal locality and makes the models sensitive to
	frame position.
- The CNN preserves temporal structure through shared convolutional filters and
	remains the preferred final reading-detector architecture.
- ANN and DNN remain useful baselines and ensemble candidates, but their
	current results do not justify replacing the CNN.

### Most promising first experiment

1. Standardize training features subject-safely.
2. Retrain ANN and DNN on the flattened standardized `(75, 10)` windows.
3. Retrain both models using statistical window summaries.
4. Apply AdamW, weight decay, scheduling, and early stopping.
5. Compare repeated-seed results against the current CNN.

## Version 3.1 — Expanded Dataset, Timestamp Standardization, and Model Evaluation

**Note:** The project now uses six subjects and 18 raw MP4 clips. The established
pipeline standardizes extracted observations to 7.5 FPS and uses approximately
five-second windows. The complete CNN, ANN, and DNN 35-epoch evaluation was
performed and recorded.

### Dataset expansion

- Dataset contains 6 subjects: `SUBJECT_01` through `SUBJECT_06`.
- Dataset contains 18 clips in total: 3 clips per subject.
- `data/features.csv` contains 77,519 usable frame-level rows.
- The timestamped dataset is `data/features_7_5fps.csv`.
- Original clips have different native frame rates, approximately from 7.5 FPS
to 30 FPS or more.
- Subject-level splitting remains required because clips from the same subject
must not appear simultaneously in training and validation.

### Standardized temporal pipeline

1. MediaPipe processes native video frames and extracts pitch, yaw, and gaze
ratio while preserving the original frame number.
2. Source FPS is read for each clip, and timestamps are calculated as
`timestamp_seconds = frame_number / source_video_fps`.
3. Timestamp-aware resampling standardizes observations to 7.5 FPS.
4. Three-class windows use 38 frames per window and a stride of 10 frames.
5. Binary reading windows use 38 frames per window and a stride of 4 frames.
6. At 7.5 FPS, 38 frames represent approximately 5 seconds.
7. Resampling retains observation gaps as `NaN`; interpolation is not allowed
across gaps larger than the configured maximum gap.
8. Short internal gaps are filled using the approved bounded-gap procedure.
9. Windows exceeding the configured missing-data fraction are rejected, and
windows with residual missing values do not enter model training.
10. Configured quality values are `max_gap_seconds: 0.40` and
`max_missing_fraction: 0.10`.

### Feature inputs and label design

- Three-class models receive only pitch, yaw, and gaze ratio with shape
approximately `(38, 3)`, without velocity or acceleration channels.
- The binary reading models receive the three base features plus velocity,
acceleration, and horizontal-motion features, producing `(38, 10)`.
- The binary reading label is mutually exclusive: `0 = not reading` and
`1 = reading`.
- The three-class activity labels are mutually exclusive: each frame/window is
assigned exactly one of Talking, Looking down, or Reading.
- The current frame/class recording structure cannot represent simultaneous
states such as Talking while Reading. That behavior requires a multilabel
recording and model design rather than mutually exclusive softmax labels.
- Both binary and three-class datasets were built from the same
`SUBJECT_06` recordings, but they use different window strides and overlap,
so their window counts and samples are not identical. The three-class
set has 754 `SUBJECT_06` windows; the binary set has 1,881 windows.

### Files added or updated

- `scripts/add_video_timestamps.py` derives clip timestamps from source FPS
without repeating landmark extraction.
- `scripts/run_evaluation.py` reports accuracy, balanced accuracy, precision,
recall, F1, log loss, multiclass or binary ROC AUC, confusion matrices, and
per-class metrics.
- `src/features/resampling.py` performs timestamp-aware resampling and preserves
unbridgeable gaps as missing values.
- `src/windowing/make_windows.py` creates standardized windows and rejects
unsafe missing-data windows.
- `src/features/motion.py` supplies reading-detector motion features.
- `configs/config.yaml` defines 7.5 FPS, 38-frame windows, gaps, and missing-data
thresholds.
- `scripts/run_training.py` and `scripts/run_reading_training.py` accept the
standardized dataset and train the requested architecture for the configured
number of epochs.
- `tests/test_windowing.py` includes tests for resampling, motion features,
short-gap filling, long-gap preservation, and window/model architecture.

### 35-epoch SUBJECT_06 evaluation

All three-class models were trained for 35 epochs with `SUBJECT_06` held out.
The models used 3,672 total generated windows across training and validation,
with 2,918 training windows and 754 validation windows for this fold.

#### Three-class CNN

- Accuracy: 68.30%
- Balanced accuracy: 68.59%
- Macro precision: 67.92%
- Macro recall: 68.59%
- Macro F1: 66.97%
- Weighted F1: 67.50%
- ROC AUC OvR macro: 85.11%
- ROC AUC OvR weighted: 85.59%
- Log loss: 1.4977
- Talking: precision 48.36%, recall 63.94%, F1 55.07%
- Looking down: precision 86.73%, recall 96.96%, F1 91.56%
- Reading: precision 68.65%, recall 44.88%, F1 54.27%
- Confusion matrix rows are actual; columns are predicted:
  `[[133, 25, 50], [0, 255, 8], [142, 14, 127]]`

#### Three-class ANN

- Accuracy: 70.69%
- Balanced accuracy: 71.35%
- Macro precision: 72.19%
- Macro recall: 71.35%
- Macro F1: 70.53%
- Weighted F1: 71.02%
- ROC AUC OvR macro: 85.60%
- ROC AUC OvR weighted: 85.98%
- Log loss: 1.1929
- Talking: precision 49.83%, recall 72.60%, F1 59.10%
- Looking down: precision 97.97%, recall 91.63%, F1 94.70%
- Reading: precision 68.78%, recall 49.82%, F1 57.79%
- Confusion matrix rows are actual; columns are predicted:
  `[[151, 3, 54], [12, 241, 10], [140, 2, 141]]`

#### Three-class DNN

- Accuracy: 72.81%
- Balanced accuracy: 74.27%
- Macro precision: 75.91%
- Macro recall: 74.27%
- Macro F1: 72.43%
- Weighted F1: 72.66%
- ROC AUC OvR macro: 89.18%
- ROC AUC OvR weighted: 89.53%
- Log loss: 0.8069
- Talking: precision 51.78%, recall 84.13%, F1 64.10%
- Looking down: precision 97.98%, recall 92.40%, F1 95.11%
- Reading: precision 77.98%, recall 46.29%, F1 58.09%
- Confusion matrix rows are actual; columns are predicted:

| Actual \ Predicted | Talking | Looking down | Reading |
|---|---:|---:|---:|
| Talking | 175 | 5 | 28 |
| Looking down | 11 | 243 | 9 |
| Reading | 152 | 0 | 131 |

### Binary reading model evaluation

Binary reading models were trained for 35 epochs and evaluated on
`SUBJECT_06` using 1,881 held-out windows and class weighting.

#### Binary CNN

- Accuracy: 53.69%
- Balanced accuracy: 46.14%
- Macro precision: 44.49%
- Macro recall: 46.14%
- Macro F1: 43.90%
- Reading precision: 28.79%
- Reading recall: 15.86%
- Reading F1: 20.46%
- ROC AUC: 65.64%
- Log loss: 5.7594
- Confusion matrix: `[[898, 277], [594, 112]]`

#### Binary ANN

- Accuracy: 62.25%
- Balanced accuracy: 56.76%
- Macro precision: 58.18%
- Macro recall: 56.76%
- Macro F1: 56.56%
- Reading precision: 49.60%
- Reading recall: 34.70%
- Reading F1: 40.83%
- ROC AUC: 72.34%
- Log loss: 1.9046
- Confusion matrix: `[[926, 249], [461, 245]]`

#### Binary DNN

- Accuracy: 58.16%
- Balanced accuracy: 55.26%
- Macro precision: 55.29%
- Macro recall: 55.26%
- Macro F1: 55.27%
- Reading precision: 44.19%
- Reading recall: 43.63%
- Reading F1: 43.91%
- ROC AUC: 66.44%
- Log loss: 0.7891
- Confusion matrix:

| Actual \ Predicted | Not Reading | Reading |
|---|---:|---:|
| Not Reading | 786 | 389 |
| Reading | 398 | 308 |

### Saved checkpoints

- Three-class checkpoints: `models/cnn_classifier_35epochs.pt`,
`models/ann_classifier_35epochs.pt`, and `models/dnn_classifier_35epochs.pt`.
- Binary checkpoints: `models/reading_cnn_35epochs_SUBJECT_06.pt`,
`models/reading_ann_35epochs_SUBJECT_06.pt`, and
`models/reading_dnn_35epochs_SUBJECT_06.pt`.
- Raw evaluation text is stored in the corresponding files under `results/`.

### Interpretation and next steps

- The three-class DNN is currently the strongest three-class subject-held-out
architecture, but the single SUBJECT_06 result is not a complete cross-subject
estimate.
- Reading is the weakest three-class class, particularly for CNN and DNN recall.
- The binary reading models are substantially weaker than the three-class
models; none provides adequate reading recall for deployment.
- Binary ANN has the best accuracy and ROC AUC in this fold; binary DNN has the
best Reading recall and F1 among binary models in the present SUBJECT_06 fold.
- Future evaluation should train and report leave-one-subject-out results across
all six subjects, with identical seeds and repeated runs where feasible.
- If overlapping behavior is meaningful, the data should be relabeled as
multilabel or behavior segments should be separated to avoid forcing
simultaneous Reading and Talking into one exclusive class.
- Improve binary detectors through better subject-safe normalization, window
summaries, regularization, calibration, class-weight tuning, and cross-subject
validation before deployment.

## Version 3.1 CNN+LSTM comparison — 35 epochs

A CNN+LSTM architecture was added for the same 38-frame, three-feature,
three-class `SUBJECT_06`-held-out task. The model uses two one-dimensional
convolution layers for local temporal feature extraction followed by a 64-unit
LSTM and a final softmax classifier. It was trained for 35 epochs on the same
2,918 training windows and evaluated on the same 754 `SUBJECT_06` windows used
by the three-class DNN comparison.

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC AUC OvR macro | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Three-class DNN | 72.81% | 74.27% | 75.91% | 74.27% | 72.43% | 72.66% | 89.18% | 0.8069 |
| Three-class CNN+LSTM | 77.72% | 78.18% | 78.12% | 78.18% | 77.38% | 77.87% | 90.67% | 0.8542 |

Compared with the DNN, CNN+LSTM increased:

- Accuracy by 4.91 percentage points.
- Balanced accuracy by 3.91 percentage points.
- Macro F1 by 4.95 percentage points.
- Macro ROC AUC by 1.49 percentage points.

The CNN+LSTM produced the following confusion matrix, with rows representing
actual classes and columns representing predicted classes:

| Actual \ Predicted | Talking | Looking down | Reading |
|---|---:|---:|---:|
| Talking | 162 | 2 | 44 |
| Looking down | 0 | 254 | 9 |
| Reading | 113 | 0 | 170 |

CNN+LSTM per-class metrics:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Talking | 58.91% | 77.88% | 67.08% |
| Looking down | 99.22% | 96.58% | 97.88% |
| Reading | 76.23% | 60.07% | 67.19% |

The CNN+LSTM improved Reading recall from 46.29% to 60.07% and Reading F1 from
58.09% to 67.19%. Looking-down classification remained the strongest class.
CNN+LSTM is therefore the stronger of the two 35-epoch `SUBJECT_06` models, but
its higher log loss (0.8542 versus 0.8069) indicates poorer probability
calibration than the DNN. The result is a single held-out-subject comparison
and should be confirmed using leave-one-subject-out evaluation and repeated
seeds before deployment.

- Model implementation: `src/model/cnn_lstm_model.py`
- Checkpoint: `models/cnn_lstm_classifier_35epochs_SUBJECT_06.pt`
- Evaluation log: `results/cnn_lstm_35epochs_subject06_eval.txt`

## Version 3.1 final temporal comparison — DNN+LSTM

A DNN+LSTM comparison model was added to the three-class 35-epoch architecture
set. Each frame is projected by dense layers (`3 -> 64 -> 64`) before a
64-unit LSTM models temporal dependencies and a 32-unit dense classifier emits
three probabilities. It was evaluated using the same 754 `SUBJECT_06` windows
and the same preprocessing and window construction as the flat DNN and
CNN+LSTM.

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | ROC AUC OvR macro | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat DNN | 72.81% | 74.27% | 75.91% | 74.27% | 72.43% | 89.18% | 0.8069 |
| DNN+LSTM | 73.08% | 73.33% | 73.23% | 73.33% | 72.38% | 89.78% | 0.8099 |
| CNN+LSTM | 77.72% | 78.18% | 78.12% | 78.18% | 77.38% | 90.67% | 0.8542 |

DNN+LSTM confusion matrix (rows=actual, columns=predicted):

| Actual \ Predicted | Talking | Looking down | Reading |
|---|---:|---:|---:|
| Talking | 145 | 13 | 50 |
| Looking down | 3 | 254 | 6 |
| Reading | 128 | 3 | 152 |

DNN+LSTM per-class metrics:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Talking | 52.54% | 69.71% | 59.92% |
| Looking down | 94.07% | 96.58% | 95.31% |
| Reading | 73.08% | 53.71% | 61.91% |

Compared with the flat DNN, DNN+LSTM improved accuracy by 0.27 points and
Reading recall by 7.42 points, but macro F1 decreased by 0.05 points. CNN+LSTM
is the strongest single-fold model: it leads flat DNN, DNN+LSTM, and the
present temporal comparison in accuracy, balanced accuracy, macro precision,
macro recall, macro F1, and macro ROC AUC. Results remain one-subject estimates
and should be confirmed by leave-one-subject-out cross-validation and repeated
seeds.

- Model implementation: `src/model/dnn_lstm_model.py`
- Checkpoint: `models/dnn_lstm_classifier_35epochs_SUBJECT_06.pt`
- Evaluation log: `results/dnn_lstm_35epochs_subject06_eval.txt`

### Forward selection: CNN+LSTM

The project will carry forward with the three-class CNN+LSTM as the primary
model candidate. This decision is based on the 35-epoch, single-subject
`SUBJECT_06` comparison:

- CNN+LSTM: 77.72% accuracy and 77.38% macro F1.
- Flat DNN: 72.81% accuracy and 72.43% macro F1.
- DNN+LSTM: 73.08% accuracy and 72.38% macro F1.

CNN+LSTM also produced the highest balanced accuracy (78.18%) and macro ROC
AUC (90.67%) in the temporal comparison, plus the strongest improvement in
Reading performance (60.07% recall and 67.19% F1). The flat DNN and DNN+LSTM
remain useful baselines for comparison, but are not the forward architecture.

Before deployment, the project will:

1. Calibrate CNN+LSTM probabilities; its log loss was 0.8542, above both
flat DNN (0.8069) and DNN+LSTM (0.8099).
2. Reduce the currently high Talking/Reading confusion through repeated-seed
training, leave-one-subject-out validation, and targeted temporal feature
or regularization experiments.
3. Retain the standardized 38-frame, 7.5 FPS, three-feature input pipeline
and subject-held-out evaluation protocol.
4. Treat the current ranking as preliminary evidence from one held-out
subject until it is confirmed across all subjects and repeated runs.
