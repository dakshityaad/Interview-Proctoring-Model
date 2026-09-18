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
