# How To Run the Interview Proctoring Project

This guide explains how to set up the project, inspect the data, extract features, train the CNN models, read validation accuracy, and run the tests.

## 1. Project goals

The project uses webcam face landmarks to classify three behaviors:

| Label | Meaning |
|---:|---|
| 0 | Normal interview behavior / not cheating |
| 1 | Looking down / possible phone use |
| 2 | Reading from a screen |

There are two CNN workflows:

1. The original three-class CNN uses 60-frame windows of `pitch`, `yaw`, and `gaze_ratio`.
2. The reading detector uses five-second windows and predicts binary `reading` versus `not reading`.

The reading detector is the recommended workflow when the requirement is to detect reading motion over approximately 4-5 seconds.

## 2. Environment setup

Use the Python 3.12 environment because MediaPipe is pinned to `0.10.35` and the project was validated with Python 3.12.

From PowerShell at the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Check the interpreter and installed packages:

```powershell
python --version
python -m pip list
```

The project root should be:

```text
D:\Interview-Proctoring-Model
```

## 3. Data layout

Raw videos should be stored under `data/raw`. The supported class names are:

```text
data/raw/
  SUBJECT_01/
    TALKING.mp4
    LOOKING.mp4
    READING.mp4
  SUBJECT_02/
    TALKING.mp4
    LOOKING.mp4
    READING.mp4
```

The extractor also accepts flat filenames such as:

```text
SUBJECT_01_TALKING.mp4
SUBJECT_01_LOOKING.mp4
SUBJECT_01_READING.mp4
```

Raw videos should contain clear examples of the intended behavior. In particular, the `LOOKING` videos should include looking down without continuously reading, because this is the main negative example for the reading detector.

## 4. Extract features

Feature extraction runs MediaPipe FaceLandmarker and saves one row per usable frame to `data/features.csv`.

Run:

```powershell
python scripts\run_extraction.py
```

The generated CSV contains:

```text
subject_id,class_label,frame_number,pitch,yaw,gaze_ratio
```

Frames without a face or valid pose are skipped. The original frame number is retained so missing detections remain visible as gaps.

## 5. Run the EDA

The EDA notebook is located at [notebooks/exploration.ipynb](notebooks/exploration.ipynb).

It examines:

- dataset size and missing values
- class balance
- subject-level class balance
- pitch, yaw, and gaze distributions
- feature correlations
- CNN window counts

The written findings are in [findings.md](findings.md).

To open the notebook from the repository root:

```powershell
code notebooks\exploration.ipynb
```

The current verified frame-level dataset contains 38,085 rows and the original CNN windowing produces 3,766 windows with the default 60-frame, stride-10 settings.

## 6. Train the original three-class CNN

The original CNN classifies normal behavior, looking down, and reading:

```powershell
python scripts\run_training.py --epochs 20
```

The default command:

- reads `data/features.csv`
- creates 60-frame windows with stride 10
- holds out the last subject alphabetically, normally `SUBJECT_03`
- trains on the remaining subjects
- validates on the held-out subject
- saves the model to `models/cnn_classifier.pt`

To choose a specific held-out subject:

```powershell
python scripts\run_training.py --test-subject SUBJECT_02 --epochs 20
```

Useful options:

```powershell
python scripts\run_training.py `
  --features-csv data\features.csv `
  --model-path models\cnn_classifier.pt `
  --test-subject SUBJECT_03 `
  --epochs 20 `
  --batch-size 64 `
  --learning-rate 0.001
```

You can also use the command on one line:

```powershell
python scripts\run_training.py --features-csv data\features.csv --model-path models\cnn_classifier.pt --test-subject SUBJECT_03 --epochs 20 --batch-size 64 --learning-rate 0.001
```

### How to read the CNN score

The command prints output similar to:

```text
Held-out subject: SUBJECT_03
Train windows:  ...; validation windows: ...
Initial loss: 0.7084
Final loss: 0.1830
Final validation accuracy: 65.30%
Saved model: ...
```

`Final validation accuracy` is calculated as:

```text
correct predictions on the held-out subject / all held-out windows
```

This is subject-level validation accuracy, which is more meaningful than randomly splitting adjacent frames from the same person.

Previously verified results were:

- `SUBJECT_03` held out: `65.30%` validation accuracy
- `SUBJECT_02` held out: `55.08%` validation accuracy

These results should be treated as small-dataset pilot results, not as a final generalization benchmark.

## 7. Train the five-second reading detector

The reading detector uses:

- 15 FPS configuration
- 75 frames, representing 5 seconds
- 0.5-second stride between windows
- original features: pitch, yaw, gaze ratio
- motion features: velocity, acceleration, and horizontal movement
- binary target: `0 = not reading`, `1 = reading`

Train it with:

```powershell
python scripts\run_reading_training.py --epochs 20
```

The default output model is:

```text
models/reading_cnn_classifier.pt
```

Choose a held-out subject explicitly:

```powershell
python scripts\run_reading_training.py --test-subject SUBJECT_02 --epochs 20
```

The command reports:

```text
Window: 75 frames (5 seconds at 15 FPS)
Held-out subject: SUBJECT_03
Train windows: ...; validation windows: ...
Class weights: [..., ...]
Final validation accuracy: ...%
Saved model: ...
```

The reading trainer uses inverse-frequency class weights because reading is currently the smaller class.

The verified dataset currently produces 4,690 binary reading windows with shape `(75, 10)`.

## 8. Understand the live reading score

The runtime detector is implemented in [src/model/reading_detector.py](src/model/reading_detector.py).

It accepts one frame at a time:

```python
from src.model.reading_detector import ReadingDetector

detector = ReadingDetector("models/reading_cnn_classifier.pt")
result = detector.update(pitch, yaw, gaze_ratio)
```

Before 75 frames have been collected, `result` is `None`. Once the five-second buffer is full, the result contains:

```python
{
    "reading_probability": 0.0,
    "is_reading": False,
}
```

The default alert policy is configured in `configs/config.yaml`:

- start reading alert at probability `>= 0.70`
- require 2 consecutive positive windows
- clear reading alert at probability `<= 0.40`
- require 2 consecutive negative windows

This prevents one accidental glance from immediately triggering an alert.

## 9. Run the tests

Run the focused reading-pipeline tests:

```powershell
python -m unittest tests.test_windowing tests.test_gaze -v
```

Expected result:

```text
Ran 4 tests ...
OK
```

The tests cover:

- motion feature shape
- five-second window construction
- binary reading labels
- persistent alert behavior
- gaze scale invariance

Python compilation checks can be run with:

```powershell
python -m py_compile src\features\motion.py src\windowing\make_windows.py src\model\train.py src\model\reading_detector.py src\scoring\reading_score.py scripts\run_reading_training.py
```

## 10. Current scoring limitations

The current training scripts report overall held-out validation accuracy. They do not yet generate a complete classification report, confusion matrix, precision, recall, F1 score, false-alert rate, or detection delay report.

Also, `scripts/run_evaluation.py` is currently a placeholder. Therefore, do not claim that a saved checkpoint has been fully evaluated unless the validation accuracy printed by the training script or a separate evaluation implementation is available.

For the reading detector, the most important future metrics are:

- reading precision
- reading recall
- false reading alerts per minute
- detection delay
- confusion between looking down and reading
- performance on a completely unseen subject

## 11. Important limitations

- The current feature CSV does not persist the source video FPS. The five-second detector therefore assumes the configured 15 FPS sampling rate.
- The dataset has only three subjects and SUBJECT_01 has no class 2 recording.
- The live Streamlit files are not wired yet: `app/video_processor.py` and `app/streamlit_app.py` are currently empty.
- The model should not be treated as proof of cheating. It detects visual behavior patterns and should be used as an alert or review signal.

## 12. Recommended run order

For a fresh run, use this order:

```powershell
# 1. Activate Python 3.12 environment
.\.venv312\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Extract or refresh features
python scripts\run_extraction.py

# 4. Inspect the EDA notebook and findings
code notebooks\exploration.ipynb

# 5. Train the original three-class CNN
python scripts\run_training.py --test-subject SUBJECT_03 --epochs 20

# 6. Train the five-second binary reading detector
python scripts\run_reading_training.py --test-subject SUBJECT_03 --epochs 20

# 7. Run regression tests
python -m unittest tests.test_windowing tests.test_gaze -v
```
