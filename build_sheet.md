# Build Sheet: Interview Screen-Reading Detector — From Scratch

This sheet explains **why each file exists, what depends on what, and what
each function needs to do** — the reasoning chain behind the build, before
any actual code is written. Use this as the blueprint to follow, in order.

---

## The Overall Approach (the reasoning chain)

The whole project is one long dependency chain — each piece only becomes
possible once the piece before it works:

```
We need to detect CHEATING BEHAVIOR
        │
        ▼ but a model can't read raw video directly in a useful way,
          so first we need NUMBERS that describe what the face is doing
        │
We need PITCH, YAW, GAZE_RATIO per frame
        │
        ▼ but we can't compute pitch/yaw/gaze without knowing where the
          face and eyes ARE in the frame
        │
We need FACIAL LANDMARKS (MediaPipe FaceMesh) per frame
        │
        ▼ landmarks only exist if we can read frames from a video/webcam
        │
We need a FRAME READER (video/webcam capture)
        │
        ▼ once we have [pitch, yaw, gaze_ratio] per frame, a SINGLE frame
          is not enough to tell "reading" from "looking down" — we need
          the pattern over TIME
        │
We need WINDOWS (sequences of frames) with LABELS attached
        │
        ▼ only now do we have something a model can actually learn from
        │
We TRAIN the CNN / DNN / ANN on these labeled windows
        │
        ▼ a trained model sitting on disk does nothing by itself
        │
We need LIVE INFERENCE (load model, run it on a live rolling window)
        │
        ▼ inference needs to be visible to a human, not just numbers
          in a terminal
        │
We need the STREAMLIT APP (webcam in browser → overlay → alert)
```

Every file below exists because of exactly one link in this chain — none
of them are "extra," and none can be skipped or reordered.

---

## File-by-File Breakdown

### `src/capture/webcam_stream.py`
**Why this file exists**: everything starts with reading frames — from a
recorded video file (during data collection/testing) or a live webcam
(during the real demo). Nothing else can run without this.

**Functions**:
- `read_video_frames(path)` — opens a video file, yields one frame at a
  time until the video ends. *Why*: your recorded subject videos need to
  be processed frame by frame for feature extraction; this is the shared
  entry point so every other script doesn't reinvent frame-reading.
- `read_webcam_frames()` — same idea, but opens the live webcam (device 0)
  instead of a file. *Why*: used later for local testing before the
  Streamlit version exists, and to sanity-check landmarks in real time.

---

### `src/landmarks/face_mesh.py`
**Why this file exists**: raw pixels mean nothing to a simple model — we
need MediaPipe's pretrained FaceMesh to tell us *where* the face, eyes,
and iris are in each frame, in the form of coordinate points.

**Functions**:
- `init_face_mesh()` — creates and configures one MediaPipe FaceMesh
  instance (with iris refinement turned on). *Why a separate function*:
  creating this is a bit expensive, so it should happen once, not per
  frame.
- `get_landmarks(face_mesh, frame)` — runs the FaceMesh instance on one
  frame, returns the list of landmark points (or `None` if no face was
  found). *Why*: this is the single choke point every other calculation
  (head pose, gaze) depends on — if this is wrong, everything downstream
  is wrong too, which is why it gets tested visually before anything else
  is built on top of it.

---

### `src/landmarks/head_pose.py`
**Why this file exists**: landmarks alone are just dots — we need to turn
6 specific dots (nose, chin, eye corners, mouth corners) into an actual
angle (pitch/yaw) describing which way the head is turned.

**Functions**:
- `get_head_pose(landmarks, frame_width, frame_height)` — picks out the 6
  needed landmark points, runs `cv2.solvePnP` against a generic 3D face
  model, and returns `(pitch, yaw)` in degrees. *Why solvePnP
  specifically*: it's the standard way to recover 3D orientation from a
  2D image when you know the rough 3D shape of the object (a face) —
  explained in the theory section of the README/report.

---

### `src/landmarks/gaze.py`
**Why this file exists**: head pose alone misses "just the eyes moved,
head didn't" — a real behavior in both looking-down and reading cases —
so we need a second, independent signal from the iris position.

**Functions**:
- `get_gaze_ratio(landmarks, frame_width, frame_height)` — finds the iris
  center and the eye corners, returns a ratio describing how far off-
  center the iris is sitting. *Why a ratio, not raw pixel position*: a
  ratio is scale-invariant — it works whether someone's face fills the
  frame or is small and far from the camera, which raw pixel coordinates
  would not.

---

### `src/features/extract_features.py`
**Why this file exists**: this is where the first three files actually
get *used together* — it loops through your recorded videos and turns
each one into a table of numbers, which is the raw material for
everything after this point.

**Functions**:
- `extract_features_from_video(path, subject_id, class_label)` — reads
  every frame (via `webcam_stream.py`), gets landmarks (via
  `face_mesh.py`), computes pitch/yaw (via `head_pose.py`) and gaze_ratio
  (via `gaze.py`), and returns a list of rows:
  `[subject_id, class_label, frame_number, pitch, yaw, gaze_ratio]`.
  *Why one row per frame*: keeping the data at frame-level (not already
  windowed) keeps this function simple and reusable — windowing is a
  separate concern, handled in its own file next.
- `build_features_csv(data_raw_dir, output_csv_path)` — walks through
  every subject folder and every class video inside `data/raw/`, calls
  `extract_features_from_video` on each, and writes everything into one
  combined `features.csv`. *Why one combined file*: makes it trivial to
  load, filter, and group by subject/class later, rather than juggling
  many small per-video files.

---

### `src/windowing/make_windows.py`
**Why this file exists**: a single frame's numbers are ambiguous — a
model needs to see a *stretch* of time to recognize the sustained
downward gaze or the reading rhythm. This file turns the flat
`features.csv` into the actual labeled training examples.

**Functions**:
- `load_features(csv_path)` — reads `features.csv` into a table, grouped
  by `(subject_id, class_label)`. *Why grouped*: windows must never cross
  a boundary between two different videos/classes — sliding a window
  across the join point between two different clips would create a
  meaningless, mixed-label window.
- `slice_into_windows(frame_sequence, window_size, stride)` — takes one
  class's frame sequence and cuts it into overlapping chunks (e.g. 30
  frames long, moving forward 5-10 frames each time). *Why overlapping*:
  with limited subjects/videos, overlapping windows multiply the number
  of usable training examples from the same footage, rather than wasting
  data with non-overlapping cuts.
- `build_dataset(csv_path, window_size, stride)` — runs the above across
  every subject/class combination, and returns the final dataset: a list
  of `(window_of_feature_vectors, class_label)` pairs. *Why this is the
  final output of this file*: this exact structure is what every model
  (CNN, DNN, ANN) will directly load and train on — nothing after this
  point touches raw video or per-frame data again.

---

### `src/model/dataset.py`
**Why this file exists**: the windows from the previous file still need
to be split responsibly (by subject, not randomly) before any training
happens, or the evaluation numbers would be misleading.

**Functions**:
- `split_by_subject(dataset, test_subject_id)` — separates the full
  dataset into train/test sets by holding out **all** windows belonging
  to one chosen subject. *Why subject-level, not random*: a random split
  could put two overlapping windows from the *same* clip into both train
  and test, letting the model "cheat" by nearly memorizing test data it
  effectively already saw — a subject-level split avoids this entirely.
- `to_tensors(dataset)` — converts the list of `(window, label)` pairs
  into the tensor format your chosen framework (PyTorch/TensorFlow)
  expects for training.

---

### `src/model/cnn_model.py`, `dnn_model.py`, `ann_model.py`
**Why these files exist, and why they're separate**: each defines one
architecture only — keeping them separate makes it trivial to swap which
model `train.py` trains without touching training logic itself, and makes
the Version 1 vs. Version 2 comparison clean (same training code, three
different architecture files).

**Functions** (same shape in each file):
- `build_model()` — defines and returns the untrained network: for the
  CNN, a few 1D-conv layers + pooling + dense + softmax; for the DNN, a
  deeper stack of fully-connected layers; for the ANN, a shallow 1-2
  layer fully-connected network. *Why separated like this*: this is the
  one place where the actual "deep learning" decisions live — everything
  before this point is feature engineering, not learning.

---

### `src/model/train.py`
**Why this file exists**: one shared training loop, reused for all three
models, so the comparison between them is fair (same optimizer, same
loss, same epochs — only the architecture changes).

**Functions**:
- `train_model(model, train_data, val_data, epochs)` — runs the standard
  training loop: forward pass → compute cross-entropy loss → backward
  pass → optimizer step, repeated per batch, per epoch, tracking
  validation loss/accuracy along the way. *Why validation tracking
  matters*: lets you notice overfitting (train accuracy climbing, val
  accuracy stalling or dropping) early, rather than only after final
  testing.
- `save_model(model, path)` — writes the trained weights to
  `models/<name>_classifier.pt` (or equivalent), so training doesn't need
  to be repeated every time you want to use or evaluate the model.

---

### `src/model/infer.py`
**Why this file exists**: separates "using a trained model" from
"training one" — this is what both your evaluation script and your live
Streamlit app will call, so it needs to be simple and fast.

**Functions**:
- `load_model(path)` — loads saved weights back into a model instance.
- `predict(model, window)` — takes one window of `[pitch, yaw,
  gaze_ratio]` values, runs it through the model, and returns
  `[P(0), P(1), P(2)]`. *Why this exact interface*: both the evaluation
  script (feeding it test windows) and the live app (feeding it the
  rolling window from the webcam) call this same function — one place to
  get inference logic right.

---

### `src/scoring/rolling_score.py`
**Why this file exists**: a single window's prediction can be noisy —
this file smooths predictions over recent time and decides when to
actually raise an alert, so the display isn't flickering between "cheating"
and "not cheating" every frame.

**Functions**:
- `update_rolling_average(history, new_prediction)` — appends the newest
  prediction to a fixed-length recent history and returns the smoothed
  average. *Why smoothing*: raises the bar for triggering an alert to
  "sustained" evidence, not "one noisy frame," matching how you defined
  both cheating behaviors as needing to be sustained/repeated in the
  first place.
- `check_alert(smoothed_probs, threshold)` — compares the smoothed
  class-1/class-2 probabilities against a threshold and returns which
  alert level (if any) to show.

---

### `app/video_processor.py`
**Why this file exists**: this is where everything gets wired together
for the live browser demo — it's the one place that calls the landmark,
model, and scoring functions together, per frame, inside
`streamlit-webrtc`'s required interface.

**Functions**:
- `recv(frame)` (the interface `streamlit-webrtc` requires) — for each
  incoming frame: get landmarks → compute pitch/yaw/gaze_ratio → append
  to the rolling window (kept as an attribute on this class, so it
  persists frame-to-frame) → once the window is full, call `predict()` →
  call `update_rolling_average()` and `check_alert()` → draw the
  percentage/alert text onto the frame → return the annotated frame.
  *Why the window has to be stored as an instance attribute here,
  specifically*: Streamlit re-runs its script on every interaction, but
  this processor object persists independently across frames — storing
  the window elsewhere (e.g. a normal Streamlit variable) would lose it
  constantly.

---

### `app/streamlit_app.py`
**Why this file exists**: the actual page — the part a person sees and
clicks on, wrapping the video processor in a usable interface.

**Functions**:
- `main()` — sets up the page layout (title, instructions), starts the
  `streamlit-webrtc` video component pointing at `video_processor.py`,
  and optionally shows a live chart of the rolling score alongside the
  video. *Why kept minimal*: all the real logic already lives in
  `video_processor.py` — this file's only job is layout and wiring, not
  computation.

---

## Additional Files (from the refined structure)

### `configs/config.yaml`
**Why this file exists**: window size, stride, alert thresholds, and file
paths were going to end up hardcoded inside multiple scripts otherwise —
centralizing them here means changing one number in one place instead of
hunting through several files, and makes running the CNN/DNN/ANN with
different settings clean and reproducible.

### `scripts/run_extraction.py`, `run_training.py`, `run_evaluation.py`
**Why these exist**: thin, runnable entry points that just call functions
already defined in `src/` — `extract_features.py` and `make_windows.py`
are libraries of functions, not something you run directly; these scripts
are what you'd actually execute from the command line, reading settings
from `configs/config.yaml`.

### `results/cnn/`, `results/dnn/`, `results/ann/`
**Why this exists**: since Phase 2's deliverable is explicitly a
comparison across three models, each model needs its own place to store
its confusion matrix, metrics table, and training curves — `run_evaluation.py`
writes here, keeping comparison outputs separate from exploratory
notebook work.

### `tests/test_landmarks.py`, `test_windowing.py`
**Why this exists**: a few small sanity checks (e.g., "does `get_head_pose`
return two floats given a fixed dummy input") catch silent breakage in the
landmark or windowing logic early — cheap insurance, not a full test suite.

---

## Order to Actually Build These In

1. `capture/webcam_stream.py` — nothing works without reading frames first.
2. `landmarks/face_mesh.py` — test visually before anything else.
3. `landmarks/head_pose.py` and `landmarks/gaze.py` — test the printed
   numbers by moving your head/eyes deliberately.
4. `features/extract_features.py` — run once on your recorded subject(s),
   produces `features.csv`.
5. `windowing/make_windows.py` — turns `features.csv` into the real
   training dataset.
6. `model/dataset.py` — subject-level split, tensor conversion.
7. `model/cnn_model.py` (then later `dnn_model.py`, `ann_model.py`).
8. `model/train.py` — train the CNN first, get *something* working
   end-to-end before worrying about DNN/ANN comparisons.
9. `model/infer.py` — load the trained model, run it on a test window.
10. `scoring/rolling_score.py` — smoothing and alert logic.
11. `app/video_processor.py` and `app/streamlit_app.py` — wire everything
    into the live demo, last, since it depends on every piece before it.

This sheet is the map. When you're ready to actually write any specific
file's code, tell me which one and I'll write it along with the
explanation of each line.
