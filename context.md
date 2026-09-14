# Project Context

## Version 0.0.0

**Project:** Interview Screen-Reading Detector
**Purpose:** Detect normal behavior, looking down, and reading during online interviews using webcam gaze and head-pose signals.

## Current Files

- `README.md` - Project problem, theory, data plan, limitations, and model phases.
- `build_sheet.md` - Planned source structure, file responsibilities, functions, and build order.
- `context.md` - This project-only summary and action log.
- `requirements.txt` - Required Python dependencies from the README.
- Project scaffold - Empty placeholders created for configs, data, source packages,
	scripts, models, results, tests, notebooks, and the application.

## Action Log

- Version 0.0.0: Added `context.md` with the current project structure and file summary.
- Version 0.0.0: Renamed `Build_Sheet (1).md` to `build_sheet.md`.
- Version 0.0.0: Created the README-based project directory and file scaffold.
- Version 0.0.0: Left scaffold files empty and populated only `requirements.txt` with
	the listed modeling, data, visualization, and Streamlit dependencies.

## Version 0.0.1

- Created the empty project files and folders based on the structure in `README.md`.
- Kept all source, test, configuration, model, data, notebook, and app files empty.
- Added the required package list to `requirements.txt`.
- Confirmed that the scaffold files contain no code.

## Version 0.0.2

- Implemented a working face-landmark prototype in `src/landmarks/face_mesh.py` using OpenCV and MediaPipe.
- Added a live webcam loop that reads frames, processes the face mesh, draws landmarks, and exits on `q`.
- Updated `requirements.txt` to include the needed webcam/media tooling for live face tracking.
- Added the MediaPipe face landmark model asset at `models/face_landmarker.task`.
- Verified the repo state and committed the recent changes with a message: `Update interview proctoring logic`.
- Pushed the latest version to the remote `main` branch on GitHub.

## Version 0.0.3

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
