# CNN EDA Findings

This document summarizes the exploratory data analysis performed on the interview-proctoring CNN dataset built from webcam-based gaze and head-pose features.

## Scope

The analysis was grounded in the project design in [README.md](README.md) and [context.md](context.md), and it was run against the extracted feature file at [data/features.csv](data/features.csv). The CNN model uses 60-frame temporal windows with 3 input features per frame:

- pitch
- yaw
- gaze_ratio

The project uses a 3-class labeling scheme:

- 0 = not cheating
- 1 = looking down / possible phone use
- 2 = reading from screen

---

## 1. Data overview

The verified dataset contains 38,085 frame-level rows across 3 subjects:

- SUBJECT_01
- SUBJECT_02
- SUBJECT_03

There are no missing values in the feature file.

### Class counts

| Class | Count | Share |
|---|---:|---:|
| 0 | 16,649 | 43.7% |
| 1 | 12,880 | 33.8% |
| 2 | 8,556 | 22.4% |

### Observation

The dataset is not perfectly balanced. Class 2 is the smallest and may need class weighting or careful evaluation metrics during training.

---

## 2. Feature summary statistics

### Overall feature stats

| Feature | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| pitch | -15.1835 | 21.3623 | -59.3932 | 38.2973 |
| yaw | -4.2616 | 13.4643 | -73.8806 | 60.4497 |
| gaze_ratio | 0.5384 | 0.0308 | -0.5854 | 1.1075 |

### Class-wise means

| Class | pitch | yaw | gaze_ratio |
|---|---:|---:|---:|
| 0 | -3.0956 | -1.5933 | 0.5470 |
| 1 | -37.1563 | -11.3745 | 0.5171 |
| 2 | -5.6278 | 1.2540 | 0.5540 |

### Class-wise standard deviations

| Class | pitch std | yaw std | gaze_ratio std |
|---|---:|---:|---:|
| 0 | 12.5355 | 12.1783 | 0.0323 |
| 1 | 15.7488 | 11.6538 | 0.0251 |
| 2 | 15.8038 | 13.9107 | 0.0136 |

### Interpretation

- Class 1 has a strongly negative average pitch, which matches the idea of looking down.
- Class 2 has a noticeably different yaw pattern and a slightly higher gaze_ratio than the baseline.
- Class 0 remains around neutral head orientation and a higher gaze_ratio than class 1.
- The gaze_ratio range is relatively tight compared with pitch and yaw, meaning gaze is a useful signal but likely not the only discriminative feature.

---

## 3. Feature correlations

Correlation among the three features:

| Feature pair | Correlation |
|---|---:|
| pitch vs yaw | 0.3790 |
| pitch vs gaze_ratio | 0.4220 |
| yaw vs gaze_ratio | 0.2353 |

### Interpretation

The features are related, but not redundant. This supports the CNN using all 3 channels together rather than relying on one feature alone.

---

## 4. Distribution by class

The class-specific distributions show the expected directional behavior:

- Looking-down behavior (class 1): more negative pitch values and lower gaze_ratio than class 0.
- Reading behavior (class 2): more horizontal motion and a distinct gaze_ratio pattern relative to the baseline.
- Normal interviewing behavior (class 0): more centered pitch/yaw values and a higher baseline gaze_ratio.

The shape of the distributions supports the hypothesis that the project is learning a combination of sustained downward head posture and temporal reading rhythm rather than random noise.

---

## 5. Window-level CNN input summary

The CNN is trained on 60-frame windows with a stride of 10, as defined in [configs/config.yaml](configs/config.yaml) and implemented in [src/windowing/make_windows.py](src/windowing/make_windows.py).

Verified window summary:

- Total windows: 3,766
- Window class counts:
  - 0: 1,649
  - 1: 1,272
  - 2: 845

### Observation

The temporal window distribution preserves the class imbalance present in the original frame-level data. Class 2 remains the least represented in the actual model training windows.

---

## 6. Subject-level balance and generalization risk

The feature CSV contains all three subjects, but not equally across classes:

| Subject | Class 0 | Class 1 | Class 2 |
|---|---:|---:|---:|
| SUBJECT_01 | 5,060 | 4,218 | 0 |
| SUBJECT_02 | 2,250 | 2,191 | 2,791 |
| SUBJECT_03 | 9,339 | 6,471 | 5,765 |

### Interpretation

- SUBJECT_01 is missing class 2 entirely, which is a major limitation for subject-independent validation.
- The dataset is realistic for a small pilot study, but it is not a strong balanced benchmark for generalization.
- Any test split should be subject-aware, as the project already intends.

This is consistent with the project goal of evaluating on a held-out subject rather than random frame-level splitting.

---

## 7. Main takeaways

1. The data is structured appropriately for a temporal CNN.
2. The feature channels carry meaningful signal separation by behavior class.
3. Class imbalance is present and should be addressed in training.
4. The strongest class separation appears in pitch for class 1, while class 2 is more subtle and likely depends on the temporal sequence pattern.
5. Subject-level imbalance could reduce generalization performance if the evaluation set is not chosen carefully.
6. The CNN is a reasonable architecture for the task because it uses time-varying head-pose and gaze information rather than a single static frame.

---

## 8. Recommended follow-up actions

- Track per-class precision/recall in validation, not just overall accuracy.
- Consider class-weighted loss because class 2 is underrepresented.
- Evaluate on a held-out subject and inspect confusion patterns by class.
- Review whether the reading pattern is cleanly separable from downward looking behavior in the temporal windows.
- If needed, test whether the model benefits from stronger sequence smoothing or longer windows.

---

## 9. Final conclusion

The EDA supports the project’s core hypothesis: the three feature channels capture meaningful behavioral differences among normal behavior, looking down, and reading. The strongest signal is in head pitch for downward-looking behavior, while the reading class likely relies more on temporal structure across the window. The main practical challenge is class imbalance and limited subject diversity, both of which should be treated as key training and evaluation concerns.

---

## 10. Five-second reading-detector accuracy

The binary five-second reading detector was trained for 20 epochs using
class-weighted loss and evaluated with subject-level holdout validation.

| Held-out subject | Validation accuracy |
|---|---:|
| SUBJECT_03 | 75.96% |
| SUBJECT_02 | 66.36% |

The average of these two measured folds is approximately **71.16%**.

The earlier result of 37.74% was from a one-epoch smoke test and should not be
treated as the final detector performance.

### Interpretation

- The detector is learning useful reading-related patterns, but performance
  varies substantially between people.
- The result is not yet a reliable benchmark because the dataset contains only
  three subjects.
- Accuracy alone is insufficient for this application. Reading precision,
  recall, F1 score, false reading alerts per minute, and detection delay should
  also be measured.

---

## 11. Recommended accuracy improvements

### Highest priority: collect more diverse data

Add more subjects and balanced examples of:

- genuine reading
- looking down without reading
- normal speaking and listening
- short glances away
- different reading speeds and pauses

The most important negative examples are looking-down clips that do not contain
a repeated reading motion.

### Correct timing and frame gaps

The detector currently assumes a 15 FPS feature stream, but the source video
FPS is not stored in `data/features.csv`. Incorrect FPS changes velocity and
acceleration values. Also, skipped face detections can create frame-number gaps
that are currently treated as if frames were consecutive.

The extraction pipeline should eventually persist the source FPS or resample
features to a known rate before calculating motion features. Derivatives should
also account for the actual frame-number difference.

### Add explicit reading-pattern features

Useful additional features include:

- left-to-right direction-change count
- number of small horizontal sweeps
- large return-sweep magnitude
- horizontal movement range
- movement periodicity or autocorrelation
- percentage of time spent moving horizontally

These features directly represent the expected reading rhythm:

```text
small movement -> small movement -> small movement -> large return sweep
```

### Tune the temporal window and training

Compare four-, five-, and six-second windows. Reading behavior may require more
than five seconds to become reliable. Also test longer training with early
stopping and save the best validation checkpoint rather than only the final
epoch.
