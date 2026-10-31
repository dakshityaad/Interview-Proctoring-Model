# Feature Dataset EDA Findings

This report was regenerated from `data/features.csv` by `scripts/run_eda.py`.
It covers the frame-level model inputs: `pitch`, `yaw`, and `gaze_ratio`.

## Data quality and coverage

- **Usable frames:** 77,519
- **Subjects:** 6
- **Activity clips represented:** 18 (all subjects have all three labels)
- **Missing values:** 0
- **Duplicate subject/class/frame rows:** 0
- **Gaze-ratio estimates outside the nominal 0–1 range:** 22 (0.028%)

### Frame balance by activity

| class_label | frames | share (%) |
| --- | --- | --- |
| Talking | 26462 | 34.14 |
| Looking down | 20970 | 27.05 |
| Reading | 30087 | 38.81 |

The dataset is moderately imbalanced: Reading is the largest class and Looking down the smallest. Use class-aware metrics and consider weighted loss when training.

The 22 out-of-range gaze estimates are rare but should be monitored; they can result from landmark jitter or partial face detection. Prefer clipping or robust scaling based on the training split only rather than silently dropping them.

### Frames by subject and activity

| subject_id | Talking | Looking down | Reading | Total |
| --- | --- | --- | --- | --- |
| SUBJECT_01 | 5060 | 4218 | 10385 | 19663 |
| SUBJECT_02 | 2250 | 2191 | 2791 | 7232 |
| SUBJECT_03 | 9339 | 6471 | 5765 | 21575 |
| SUBJECT_04 | 2508 | 1054 | 2388 | 5950 |
| SUBJECT_05 | 2272 | 1029 | 2401 | 5702 |
| SUBJECT_06 | 5033 | 6007 | 6357 | 17397 |

Subject coverage varies from **5,702** usable frames for SUBJECT_05 to **21,575** for SUBJECT_03. This makes subject-held-out validation preferable to random frame splitting.

### Source frame-rate caveat

Frame count is **not** a duration measure in this dataset. The clips for subjects 2, 4, and 5 were recorded at about 7.5 FPS; subject 6 at about 16.7 FPS; subject 3 at 30 FPS; and subject 1 contains clips at about 16.7 FPS plus a 30-FPS Reading clip.

| subject_id | source_fps | recorded_minutes |
| --- | --- | --- |
| SUBJECT_01 | 16.68, 30.00 | 15.61 |
| SUBJECT_02 | 7.51 | 16.1 |
| SUBJECT_03 | 30.00 | 12.61 |
| SUBJECT_04 | 7.51, 7.66 | 15.83 |
| SUBJECT_05 | 7.51, 7.62 | 15.03 |
| SUBJECT_06 | 16.68, 16.69 | 17.57 |

This has two direct implications:

1. Do not use raw frame totals to compare how much behavior was recorded per subject; compare duration instead.
2. The existing motion features and five-second-window logic assume 15 FPS, so they currently represent different amounts of real time across subjects. Resample each recording to one common FPS (for example 15 FPS) or persist per-frame time/FPS and use time-aware derivatives before training the reading detector.

## Feature summaries

### Overall values

| index | mean | std | min | max |
| --- | --- | --- | --- | --- |
| pitch | -16.0034 | 20.438 | -59.3932 | 42.1052 |
| yaw | -1.6004 | 12.1052 | -73.8806 | 74.3929 |
| gaze_ratio | 0.5343 | 0.0303 | -0.5854 | 1.4421 |

### Mean feature values by activity

| activity | pitch | yaw | gaze_ratio |
| --- | --- | --- | --- |
| Talking | -4.7058 | -0.6893 | 0.5452 |
| Looking down | -36.5879 | -7.9798 | 0.5191 |
| Reading | -11.5927 | 2.0445 | 0.5354 |

### Feature variability by activity (standard deviation)

| activity | pitch | yaw | gaze_ratio |
| --- | --- | --- | --- |
| Talking | 14.0672 | 12.1464 | 0.0327 |
| Looking down | 20.2011 | 11.7384 | 0.0273 |
| Reading | 13.8107 | 10.4586 | 0.0253 |

### Correlation

| index | pitch | yaw | gaze_ratio |
| --- | --- | --- | --- |
| pitch | 1.0 | 0.2141 | 0.3281 |
| yaw | 0.2141 | 1.0 | 0.0699 |
| gaze_ratio | 0.3281 | 0.0699 | 1.0 |

Pitch, yaw, and gaze ratio are correlated but not duplicates, so retaining all three model inputs is justified. The distribution plots show that Looking down is most clearly separated by head pitch; Reading overlaps more with the other behaviors and benefits from temporal information.

## Window-level checks

The three-class CNN configuration uses 60-frame windows with a stride of 10.

| Activity | 60-frame windows |
|---|---:|
| Talking | 2,614 |
| Looking down | 2,064 |
| Reading | 2,976 |

- **Total 60-frame windows:** 7,654

The reading detector uses 75-frame windows (5 seconds at 15 FPS) with a stride of 8.

| Binary target | 75-frame windows |
|---|---:|
| Not reading | 5,823 |
| Reading | 3,708 |

- **Total five-second windows:** 9,531

## Key takeaways

1. The refreshed dataset now contains six subjects and complete coverage for all three activities per subject.
2. No missing values or duplicate frame keys were found.
3. Raw frame counts are confounded by source FPS. Normalize sampling before using temporal or motion-based models.
4. Class and subject imbalance remain important: evaluate with per-class precision/recall/F1 and a subject-held-out split.
5. Frame-level pitch offers the clearest signal for Looking down; temporal windows remain especially important for identifying Reading.

## 35-epoch subject-held-out evaluation (Subject 06)

Models were retrained with 35 epochs using the timestamp-standardized 7.5 FPS, 38-frame dataset and evaluated on SUBJECT_06.

### Three-class activity models

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC AUC OvR (macro) | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 68.30% | 68.59% | 67.92% | 68.59% | 66.97% | 67.50% | 85.11% | 1.4977 |
| ANN | 70.69% | 71.35% | 72.19% | 71.35% | 70.53% | 71.02% | 85.60% | 1.1929 |
| DNN | 72.81% | 74.27% | 75.91% | 74.27% | 72.43% | 72.66% | 89.18% | 0.8069 |

Three-class per-label F1:

| Model | Talking | Looking down | Reading |
|---|---:|---:|---:|
| CNN | 55.07% | 91.56% | 54.27% |
| ANN | 59.10% | 94.70% | 57.79% |
| DNN | 64.10% | 95.11% | 58.09% |

### Binary reading models

| Model | Accuracy | Balanced accuracy | Reading precision | Reading recall | Reading F1 | Macro F1 | ROC AUC | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 53.69% | 46.14% | 28.79% | 15.86% | 20.46% | 43.90% | 65.64% | 5.7594 |
| ANN | 62.25% | 56.76% | 49.60% | 34.70% | 40.83% | 56.56% | 72.34% | 1.9046 |
| DNN | 58.16% | 55.26% | 44.19% | 43.63% | 43.91% | 55.27% | 66.44% | 0.7891 |

Within the original flat CNN/ANN/DNN comparison, the three-class DNN is the strongest activity classifier. Among binary reading detectors, ANN has the best accuracy and ROC AUC, while DNN has the best Reading recall and F1. The binary reading models remain weaker and should be improved before deployment, especially because Reading recall is below 50% for every binary model.

### CNN+LSTM comparison with the three-class DNN

A CNN+LSTM was trained for 35 epochs on the same 38-frame, three-feature, three-class windows and evaluated on the same 754 `SUBJECT_06` windows.

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC AUC OvR macro | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DNN | 72.81% | 74.27% | 75.91% | 74.27% | 72.43% | 72.66% | 89.18% | 0.8069 |
| CNN+LSTM | 77.72% | 78.18% | 78.12% | 78.18% | 77.38% | 77.87% | 90.67% | 0.8542 |

CNN+LSTM confusion matrix (rows=actual, columns=predicted):

| Actual \ Predicted | Talking | Looking down | Reading |
|---|---:|---:|---:|
| Talking | 162 | 2 | 44 |
| Looking down | 0 | 254 | 9 |
| Reading | 113 | 0 | 170 |

CNN+LSTM improved accuracy by 4.91 percentage points, balanced accuracy by 3.91 points, macro F1 by 4.95 points, and macro ROC AUC by 1.49 points. Reading recall increased from 46.29% to 60.07%, and Reading F1 increased from 58.09% to 67.19%. Its log loss was higher, so calibration should be checked before deployment.

### DNN+LSTM comparison with the flat DNN and CNN+LSTM

The DNN+LSTM projects each frame through dense layers before temporal modeling by a 64-unit LSTM. It was trained for 35 epochs and evaluated on the same 754 `SUBJECT_06` windows.

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC AUC OvR macro | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Flat DNN | 72.81% | 74.27% | 75.91% | 74.27% | 72.43% | 72.66% | 89.18% | 0.8069 |
| CNN+LSTM | 77.72% | 78.18% | 78.12% | 78.18% | 77.38% | 77.87% | 90.67% | 0.8542 |
| DNN+LSTM | 73.08% | 73.33% | 73.23% | 73.33% | 72.38% | 73.01% | 89.78% | 0.8099 |

DNN+LSTM confusion matrix (rows=actual, columns=predicted):

| Actual \ Predicted | Talking | Looking down | Reading |
|---|---:|---:|---:|
| Talking | 145 | 13 | 50 |
| Looking down | 3 | 254 | 6 |
| Reading | 128 | 3 | 152 |

DNN+LSTM per-class F1: Talking 59.92%, Looking down 95.31%, Reading 61.91%. Relative to the flat DNN, it improved accuracy by 0.27 points and Reading recall by 7.42 points, but reduced macro F1 by 0.05 points and macro precision by 2.68 points. CNN+LSTM remains the strongest of the three temporal-comparison architectures on this held-out subject.

## Selected model: CNN+LSTM

The project will carry forward with the three-class CNN+LSTM because it produced the best subject-held-out accuracy, balanced accuracy, macro F1, and macro ROC AUC among the three-class temporal architectures evaluated on `SUBJECT_06`. The flat DNN and DNN+LSTM remain documented comparison baselines; neither is currently the production candidate.

Carry-forward priorities for CNN+LSTM:

1. Improve probability calibration because its 0.8542 log loss is higher than the flat DNN and DNN+LSTM baselines.
2. Run repeated-seed training and leave-one-subject-out evaluation before treating the single-subject result as robust.
3. Preserve the standardized 38-frame/7.5 FPS three-feature pipeline in subsequent experiments.
4. Focus next improvements on Reading recall and Talking-versus-Reading confusion, especially the 113 Reading windows predicted as Talking and 44 Talking windows predicted as Reading in the held-out confusion matrix.
5. Keep the mutually exclusive three-class interpretation explicit unless a reliable multilabel annotation strategy is introduced.

## Generated figures

- `results/pitch_eda.png`
- `results/yaw_eda.png`
- `results/gaze_ratio_eda.png`
- `results/subject_class_coverage_eda.png`
- `results/feature_correlation_eda.png`
