# CNN Models

This document describes the convolutional neural network models used in the Interview Proctoring project.

The project currently has two CNN configurations built from the same architecture:

1. The original three-class CNN for normal behavior, looking down, and reading.
2. The five-second binary reading CNN for reading versus not reading.

The implementation is in [src/model/cnn_model.py](../src/model/cnn_model.py).

## 1. CNN purpose

The CNN receives a temporal sequence of face-behavior features rather than a raw image.

The original feature vector for each frame is:

```text
[pitch, yaw, gaze_ratio]
```

The five-second reading detector adds temporal motion channels:

```text
[pitch, yaw, gaze_ratio,
 pitch_velocity, yaw_velocity, gaze_velocity,
 pitch_acceleration, yaw_acceleration, gaze_acceleration,
 horizontal_motion]
```

The CNN is useful because reading is a time-dependent behavior. A single frame may show a person looking down, but a sequence can show repeated horizontal eye/head movement and return sweeps.

## 2. Shared architecture

The model is a 1D temporal CNN. Its input is shaped as:

```text
(batch_size, frames, input_features)
```

Before convolution, the model transposes the input to:

```text
(batch_size, input_features, frames)
```

This allows each feature to act as a convolution channel and the frame axis to act as the temporal dimension.

### Layer sequence

| Layer | Configuration | Purpose |
|---|---|---|
| Conv1d | `input_features -> 32`, kernel `5`, padding `2` | Detect short temporal patterns |
| ReLU | activation | Add non-linearity |
| MaxPool1d | kernel `2` | Downsample the time axis |
| Conv1d | `32 -> 64`, kernel `5`, padding `2` | Learn richer temporal patterns |
| ReLU | activation | Add non-linearity |
| MaxPool1d | kernel `2` | Further temporal downsampling |
| Conv1d | `64 -> 64`, kernel `3`, padding `1` | Refine local temporal features |
| ReLU | activation | Add non-linearity |
| AdaptiveAvgPool1d | output size `1` | Reduce the complete sequence to one feature vector |
| Flatten | `64` values | Prepare for dense layers |
| Linear | `64 -> 32` | Classification representation |
| ReLU | activation | Add non-linearity |
| Linear | `32 -> number_of_classes` | Produce class logits |
| Softmax | dimension `1` | Produce class probabilities |

There is currently no dropout, batch normalization, recurrent layer, attention layer, or transformer layer.

## 3. Original three-class CNN

### Configuration

| Setting | Value |
|---|---|
| Input features | `3` |
| Input channels | `pitch`, `yaw`, `gaze_ratio` |
| Window size | `60` frames |
| Window stride | `10` frames |
| Number of classes | `3` |
| Class 0 | Normal interview behavior / not cheating |
| Class 1 | Looking down / possible phone use |
| Class 2 | Reading from screen |
| Default epochs | `20` |
| Default batch size | `64` |
| Default learning rate | `0.001` (`1e-3`) |
| Optimizer | Adam |
| Loss | Negative log likelihood over `log(probabilities)` |
| Default checkpoint | `models/cnn_classifier.pt` |

### Parameter count

The original model has **25,347 trainable parameters**.

Parameter breakdown:

| Layer | Parameters |
|---|---:|
| First convolution | 512 |
| Second convolution | 10,304 |
| Third convolution | 12,352 |
| First linear layer | 2,080 |
| Output linear layer | 99 |
| **Total** | **25,347** |

### Training command

```powershell
python scripts\run_training.py --epochs 20
```

To select the held-out subject:

```powershell
python scripts\run_training.py --test-subject SUBJECT_03 --epochs 20
```

The split is subject-level. All windows belonging to the selected subject are used for validation, while the remaining subjects are used for training.

### Previously measured results

| Held-out subject | Validation accuracy |
|---|---:|
| SUBJECT_03 | 64.41% in the latest run |
| SUBJECT_03 | 65.30% in an earlier recorded run |
| SUBJECT_02 | 55.08% in an earlier recorded run |

The result can vary between runs because the model initialization and minibatch order are not currently seeded.

## 4. Five-second binary reading CNN

This model uses the same CNN architecture but changes the input and output sizes.

### Configuration

| Setting | Value |
|---|---|
| Input features | `10` |
| Window duration | `5` seconds |
| Assumed sampling rate | `15 FPS` |
| Window size | `75` frames |
| Window stride | `8` frames, approximately `0.5` seconds |
| Number of classes | `2` |
| Class 0 | Not reading; original classes 0 and 1 combined |
| Class 1 | Reading; original class 2 |
| Default epochs | `20` |
| Default batch size | `64` |
| Default learning rate | `0.001` (`1e-3`) |
| Optimizer | Adam |
| Loss | Weighted negative log likelihood |
| Default checkpoint | `models/reading_cnn_classifier.pt` |

The seven additional motion features are calculated in [src/features/motion.py](../src/features/motion.py):

- pitch velocity
- yaw velocity
- gaze velocity
- pitch acceleration
- yaw acceleration
- gaze acceleration
- horizontal motion

### Parameter count

The binary reading model has **26,434 trainable parameters**.

The increase from the original model comes from the first convolution receiving 10 input channels instead of 3 and the final layer producing 2 outputs instead of 3.

| Layer | Parameters |
|---|---:|
| First convolution | 1,632 |
| Second convolution | 10,304 |
| Third convolution | 12,352 |
| First linear layer | 2,080 |
| Output linear layer | 66 |
| **Total** | **26,434** |

### Class weighting

The reading trainer calculates inverse-frequency weights from the training windows:

```text
weight_for_class = total_training_samples / (number_of_classes * class_count)
```

This gives the smaller reading class more influence during training and reduces the chance that the model predicts not-reading for most windows.

### Training command

```powershell
python scripts\run_reading_training.py --epochs 20
```

To choose a held-out subject:

```powershell
python scripts\run_reading_training.py --test-subject SUBJECT_03 --epochs 20
```

The model is saved to:

```text
models/reading_cnn_classifier.pt
```

### Measured results

| Held-out subject | Validation accuracy |
|---|---:|
| SUBJECT_03 | 75.96% |
| SUBJECT_02 | 66.36% |
| Average of these two folds | approximately 71.16% |

The previous `37.74%` result was from a one-epoch smoke test and is not the final reading-detector result.

## 5. Runtime reading behavior

The stateful runtime wrapper is implemented in [src/model/reading_detector.py](../src/model/reading_detector.py).

It accumulates 75 frames before making its first prediction:

```python
from src.model.reading_detector import ReadingDetector

detector = ReadingDetector("models/reading_cnn_classifier.pt")
result = detector.update(pitch, yaw, gaze_ratio)
```

The result contains:

```python
{
    "reading_probability": 0.0,
    "is_reading": False,
}
```

The alert scorer uses hysteresis:

| Rule | Default |
|---|---:|
| Start reading evidence | probability `>= 0.70` |
| Consecutive positive windows | `2` |
| Stop reading evidence | probability `<= 0.40` |
| Consecutive negative windows | `2` |

This prevents one high-probability window from immediately creating a reading alert.

## 6. Training behavior

The shared training loop is in [src/model/train.py](../src/model/train.py).

For each epoch:

1. The model is put into training mode.
2. Training windows are loaded in shuffled minibatches.
3. The model produces class probabilities.
4. Log probabilities are passed to `NLLLoss`.
5. Adam updates the model parameters.
6. The model is evaluated on the held-out subject.
7. Validation loss and validation accuracy are recorded.

The training scripts currently save the final model after the requested number of epochs. They do not yet save the best validation checkpoint automatically.

## 7. Important limitations

- The original and reading models are trained on a very small number of subjects.
- `SUBJECT_01` does not contain a class 2 reading recording.
- The reading model assumes 15 FPS because the source FPS is not persisted in `data/features.csv`.
- Skipped face detections can create frame gaps that are not yet included in derivative calculations.
- The current scripts report overall validation accuracy but do not yet generate precision, recall, F1, confusion matrices, false-alert rate, or detection delay.
- The Streamlit application files are not yet wired to the runtime detector.

## 8. Recommended improvements

For higher reading-detection accuracy, prioritize:

1. Add more subjects and balanced reading/not-reading clips.
2. Store or infer the true video FPS before calculating motion features.
3. Account for frame-number gaps when calculating velocity and acceleration.
4. Add explicit left-to-right sweep, return-sweep, direction-change, and periodicity features.
5. Save the best validation checkpoint with early stopping.
6. Evaluate precision, recall, F1, false alerts per minute, and detection delay in addition to accuracy.
