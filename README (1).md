# Interview Screen-Reading Detector

A deep learning project to detect whether a candidate in an online interview is
reading their answers off-screen (a proxy for cheating), using webcam-based
gaze and head-pose tracking.

---

## 1. Problem Statement

Detect two behavioral signatures during a live interview:

1. **Looking down** — sustained downward head/gaze, not necessarily reading.
2. **Reading (cheating)** — a genuine reading rhythm: progressive small
   left-to-right (or line-by-line) eye movements followed by a large return
   sweep, repeated over time.

Everything else (speaking, listening, natural gaze at the camera) is the
**not cheating** baseline class.

This is framed as a **3-class classification problem**, where classes 1 and
2 are both cheating, at different severity/confidence levels:

| Class | Description | Verdict |
|---|---|---|
| 0 | Not cheating — speaking / listening / natural gaze | Not cheating |
| 1 | Looking down — possible phone use | Cheating (alert) |
| 2 | Reading from screen — genuine reading pattern | Cheating (stronger alert) |

---

## 2. Core Theory / Principles

- **Head pose (pitch/yaw)** is estimated via `solvePnP`, using a generic 3D
  face model matched against 2D facial landmarks (nose, chin, eye corners,
  mouth corners). Robust to lighting and largely unaffected by glasses.
- **Gaze ratio** is estimated from iris position relative to eye corners
  (MediaPipe iris landmarks). More sensitive to glasses/glare than head pose.
- **Static vs. periodic signal**: "looking down" is a *static/sustained*
  deviation, detectable from short windows. "Reading" is a *periodic*
  signature (sawtooth pattern in horizontal gaze position: small steps one
  direction, then a big jump back) — this needs a **temporal window**, not a
  single frame, to detect reliably.
- **Why feature-based, not raw-pixel end-to-end**: MediaPipe FaceMesh is a
  pretrained deep model used as a fixed feature extractor. Training a
  classifier on its output features (`[pitch, yaw, gaze_ratio]` per frame)
  needs far less data than training a CNN directly on raw video/eye-crop
  images, which matters given the realistic data budget (4-5 subjects).
- **Known confound**: note-taking on paper looks behaviorally similar to
  reading off a screen (downward gaze). Not solved architecturally — handled
  by making sure "looking down" training clips include non-reading downward
  behavior as a contrast to genuine reading clips.
- **Glasses**: degrade gaze_ratio quality (glare/occlusion) but not head
  pose. Current scope: recording **without glasses**, noted as a stated
  limitation rather than solved.
- **No public dataset fits this task** — existing gaze/reading eye-tracking
  datasets (GazeCapture, MPIIGaze, Provo Corpus, etc.) are either unlabeled
  for this behavior or collected with lab-grade eye trackers, not consumer
  webcams. Self-collected data is the realistic path.
- **Audio is out of scope** — the task is purely visual (gaze/head pose);
  audio-based hesitation analysis is a possible future direction, not part
  of the current build.

---

## 3. Data Collection Plan

- **Subjects**: 4-5 people.
- **Exactly 3 samples (videos) per subject — one per class, no more**:
  1. **Explaining normally** — talking/answering as in a real interview
     (baseline, class 0 — not cheating)
  2. **Reading from screen** — genuine continuous reading while looking at
     a screen/phone (class 2 — cheating, stronger alert)
  3. **Looking down at phone** — downward gaze, deliberately *not* reading
     continuously (class 1 — cheating, softer alert)
- **Fixed sample budget**: with only 3 samples per subject (not multiple
  takes per class), each video needs to be long enough on its own to
  yield sufficient training windows after slicing — see length note below.
  There is no redundancy per class per subject, so quality of each single
  take matters more than with a multi-take plan.
- **Length per video**: ~2-4 minutes each (longer than a large-subject-count
  plan would need, to compensate for both fewer subjects and only one
  sample per class, while still getting enough windows after slicing).
- **Recording notes**:
  - Avoid direct overhead light / backlighting (reduces iris-tracking glare).
  - "Looking down at phone" should stay clearly distinct from "reading from
    screen" — a downward stare without a sustained scan rhythm, vs.
    genuinely reading real sentences continuously — since with only one
    take per class, there's no second chance to fix an ambiguous sample.
  - No glasses for this data collection pass (stated limitation, not solved).
- **Train/test split**: hold out **one full subject** (all 3 of their
  samples) for testing; train on the remaining subjects. Subject-level
  split, not clip-level, to actually test generalization to a new person.

---

## 4. Phases & Model Versions

The project is delivered in two versions, matching Phase 1 and Phase 2 of
the work:

### Version 1 (Phase 1) — Primary model
- **Model**: 1D-CNN over sliding windows of `[pitch, yaw, gaze_ratio]`.
- **Goal**: working end-to-end pipeline — webcam → FaceMesh → CNN →
  rolling alert UI. This is the main deliverable model.

### Version 2 (Phase 2) — Comparison models
Two additional classical deep learning models trained on the same windowed
feature data (flattened), for a comparison/ablation study against the
Version 1 CNN — kept intentionally simple, no recurrent/attention
architectures:

- **Model 2: DNN (Deep Neural Network)** — a fully-connected network with
  several hidden layers (e.g. 3-4), taking a flattened window of
  `[pitch, yaw, gaze_ratio]` values as input. Tests whether a deeper plain
  network can learn the pattern without any convolution or sequence
  structure.
- **Model 3: ANN (shallow Artificial Neural Network)** — a fully-connected
  network with just 1-2 hidden layers, same flattened input. The simplest
  possible baseline — tests how much accuracy is gained by adding depth
  (DNN) or local pattern detection (CNN) over this minimal model.

**Comparison metrics across all 3 models**: per-class accuracy,
precision/recall, a collapsed binary metric (cheating = class 1 or 2, vs.
not cheating = class 0), parameter count, and training time.

---

## 5. Pipeline / Workflow

### Offline: data → trained model

```
Raw video (per subject, per class)
        │
        ▼
MediaPipe FaceMesh (pretrained) — extract per-frame:
   [pitch, yaw, gaze_ratio]
        │
        ▼
Slice into overlapping windows (e.g. 30-60 frames, stride 5-10)
        │
        ▼
Labeled dataset: (window of feature vectors, class label)
        │
        ▼
Train models (Version 1: CNN. Version 2: + DNN, + ANN)
        │
        ▼
Saved weights (models/*.pt)
```

### Online: live webcam → web app

```
Browser (user's webcam)
        │  (getUserMedia via streamlit-webrtc, HTTPS)
        ▼
Streamlit app (server-side)
        │
        ▼
streamlit-webrtc VideoProcessor — runs PER FRAME:
   ├─ MediaPipe FaceMesh → [pitch, yaw, gaze_ratio]
   ├─ append to rolling window (kept as processor instance state,
   │  NOT Streamlit script-rerun state)
   ├─ once window is full → feed into trained model (CNN, or
   │  selected Version 2 model: DNN / ANN) → [P(0), P(1), P(2)]
   ├─ smooth over recent predictions (rolling_score.py)
   └─ draw overlay (percentage + alert text) onto the frame
        │
        ▼
Annotated video frame streamed back to browser
        │
        ▼
Streamlit UI: live video panel + rolling score chart + alert banner
```

---

## 6. Project Structure

```
reading-detector/
│
├── data/
│   ├── raw/                  # raw recorded video clips, per subject/class
│   ├── frames/               # extracted frames (if needed for inspection)
│   └── features.csv          # extracted [pitch, yaw, gaze_ratio, label, subject_id, window_id]
│
├── src/
│   ├── capture/
│   │   └── webcam_stream.py       # local/offline camera capture loop (used for data collection)
│   ├── landmarks/
│   │   ├── face_mesh.py           # MediaPipe FaceMesh wrapper
│   │   ├── head_pose.py           # pitch/yaw estimation (solvePnP)
│   │   └── gaze.py                # iris/gaze ratio estimation
│   ├── features/
│   │   └── extract_features.py    # video -> per-frame feature extraction
│   ├── windowing/
│   │   └── make_windows.py        # slice feature sequences into labeled windows
│   ├── model/
│   │   ├── dataset.py             # loads windows into train/val/test splits
│   │   ├── cnn_model.py           # Version 1: 1D-CNN architecture (3-class softmax)
│   │   ├── dnn_model.py           # Version 2: deeper fully-connected network (DNN)
│   │   ├── ann_model.py           # Version 2: shallow fully-connected network (ANN)
│   │   ├── train.py               # training loop (shared across all 3 models)
│   │   └── infer.py               # loads trained weights, runs inference on a window
│   └── scoring/
│       └── rolling_score.py       # rolling window smoothing + alert logic
│
├── models/
│   ├── cnn_classifier.pt          # Version 1 saved weights
│   ├── dnn_classifier.pt          # Version 2 saved weights
│   └── ann_classifier.pt          # Version 2 saved weights
│
├── notebooks/
│   └── exploration.ipynb          # data checks, threshold/architecture experiments
│
├── app/
│   ├── streamlit_app.py           # Streamlit entry point — page layout, sidebar, charts
│   └── video_processor.py         # streamlit-webrtc VideoProcessor: per-frame
│                                   #   FaceMesh -> window -> model -> overlay
│
├── requirements.txt
└── README.md                      # this file
```

---

## 7. Model Spec

### Version 1: CNN (primary model)
- **Input**: window of shape `(30-60 frames, 3 features)` —
  `[pitch, yaw, gaze_ratio]` per frame.
- **Architecture**: 2-3 1D-conv layers (e.g. 16 → 32 → 64 filters, kernel
  size 3-5) → global pooling → 1-2 dense layers → 3-way softmax output.
- **Parameter count**: small, likely well under 100K params.

### Version 2: DNN (comparison model)
- **Input**: same window, **flattened** into a single vector (e.g. 60
  frames × 3 features = 180 values).
- **Architecture**: fully-connected network, 3-4 hidden layers (e.g.
  128 → 64 → 32 → 16), ReLU activations, dropout between layers → 3-way
  softmax output.

### Version 2: ANN (simplest baseline)
- **Input**: same flattened window vector as the DNN.
- **Architecture**: fully-connected network, 1-2 hidden layers (e.g.
  64 → 16), ReLU activation → 3-way softmax output.

**Shared across all three**: cross-entropy loss, Adam optimizer,
subject-level train/val/test split, class weighting to counter the
"not cheating" class dominating raw recorded time.

**Training compute**: CPU is sufficient for all three; no GPU strictly
required given the small input dimensionality and dataset size.

---

## 8. Requirements / Libraries

**Modeling / data (offline)**
- `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scikit-learn`
- `torch` or `tensorflow` (pick one) — CNN, DNN, ANN
- `scipy` — optional, for any additional signal-processing features you
  choose to engineer
- `matplotlib`/`seaborn`, `tqdm` — optional, for plots and progress bars

**Web deployment**
- `streamlit` — app framework / page layout
- `streamlit-webrtc` — browser webcam capture + server-side per-frame
  processing, over HTTPS

**Hardware**
- CPU is sufficient for training and for live inference; GPU only speeds
  things up, not required at this data/model scale.
- Standard webcam, ideally 15-20+ fps so the reading-saccade pattern isn't
  under-sampled.

## 9. Resources / References (for background reading)

- MediaPipe Face Mesh (landmark + iris tracking model used as the fixed
  feature extractor).
- `cv2.solvePnP` — head pose estimation from 2D-3D landmark correspondence.
- Background reading on reading eye-movement patterns (saccades, fixations,
  return sweeps) — useful for understanding *why* the saccade signature
  looks the way it does, even though the referenced lab datasets (Provo
  Corpus, GECO, Dundee Corpus) aren't directly usable as training data here.

---

## 10. Open Items / Stated Limitations

- No glasses-wearing subjects in this data collection pass — flagged as a
  known limitation, not solved.
- Note-taking vs. screen-reading confound is only partially addressed via
  the "looking down" class contrast, not fully resolved.
- No audio signal used — visual-only pipeline by design.
- Currently scoped for 4-5 subjects; more subjects/sessions would improve
  generalization if time allows later.
- Phase 2 (Version 2) models — DNN and ANN — are planned per instructor
  requirement, not yet built. Kept as classical fully-connected
  architectures (no recurrent/attention models) to keep scope manageable.
