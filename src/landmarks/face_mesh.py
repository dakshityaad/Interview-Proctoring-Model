import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import time

# Head pose and gaze are separate estimators; this module only handles the
# MediaPipe face-mesh pipeline and visualization.
from src.landmarks import head_pose
from src.landmarks.gaze import LEFT_IRIS, RIGHT_IRIS, get_gaze_ratio
from src.capture.webcam_stream import read_webcam_frames

try:
    import mediapipe as mp
except ImportError as exc:
    raise RuntimeError(
        "MediaPipe is not installed. Run:\n"
        "py -3.12 -m venv .venv312\n"
        ".\\.venv312\\Scripts\\python.exe -m pip install -r requirements.txt"
    ) from exc


def draw_gaze_overlay(frame, face_landmarks):
    """Draw iris markers and a compact gaze-position indicator."""
    height, width = frame.shape[:2]
    if len(face_landmarks) <= LEFT_IRIS:
        return

    # The estimator returns normalized positions; map them onto two display
    # bars centered at 0.5 for a straight-ahead gaze.
    gx, gy = get_gaze_ratio(face_landmarks, width, height)
    for iris_id in (RIGHT_IRIS, LEFT_IRIS):
        center = (
            int(face_landmarks[iris_id].x * width),
            int(face_landmarks[iris_id].y * height),
        )
        cv2.circle(frame, center, 3, (0, 255, 255), -1)

    bar_x, bar_y = 140, 60
    cv2.rectangle(frame, (bar_x - 50, bar_y - 6), (bar_x + 50, bar_y + 6), (255, 255, 255), 1)
    cv2.circle(frame, (bar_x + int((gx - 0.5) * 100), bar_y), 5, (0, 0, 255), -1)
    cv2.rectangle(frame, (bar_x - 6, bar_y - 30), (bar_x + 6, bar_y + 70), (255, 255, 255), 1)
    cv2.circle(frame, (bar_x, bar_y + 20 + int((gy - 0.5) * 100)), 5, (0, 0, 255), -1)


def draw_face_pose(frame, face_landmarks):
    """Draw a lightweight head-pose overlay using the shared pose helper."""
    height, width = frame.shape[:2]
    # head_pose converts the same normalized landmarks into pitch and yaw.
    pitch, yaw = head_pose.get_head_pose(face_landmarks, width, height)
    if pitch is None or yaw is None:
        return

    cv2.putText(
        frame,
        f"P {pitch:.1f} Y {yaw:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


# MediaPipe 0.10 supports either the legacy Solutions API or the newer Tasks
# API. Select the available backend once when this module is imported.
legacy_face_mesh = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")

if legacy_face_mesh:
    # The legacy backend returns a face directly from process().
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
else:
    try:
        # The Tasks backend needs an on-disk model and monotonically increasing
        # timestamps because it runs in VIDEO mode.
        from mediapipe import Image as MpImage, ImageFormat
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        model_path = "models/face_landmarker.task"
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
        )
        face_landmarker = FaceLandmarker.create_from_options(options)
    except Exception as exc:
        raise RuntimeError(
            "Your MediaPipe install is incompatible with this project. "
            "Use Python 3.12 and reinstall dependencies with:\n"
            "py -3.12 -m venv .venv312\n"
            ".\\.venv312\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc


def init_face_mesh():
    """Return the configured face-landmark detector.

    MediaPipe initialization is relatively expensive, so callers should create
    one detector and reuse it for all frames in a video.
    """
    if legacy_face_mesh:
        return face_mesh
    return face_landmarker


def get_landmarks(detector, frame):
    """Return the first face's landmarks for one BGR frame, or ``None``.

    OpenCV supplies BGR images, while both MediaPipe backends expect RGB. The
    conversion stays here so video extraction and the live demo use the same
    landmark interface.
    """
    if legacy_face_mesh:
        # The legacy API returns a result object containing detected faces.
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    # The Tasks API needs an explicit image wrapper and increasing timestamps.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
    timestamp_ms = getattr(detector, "_extraction_timestamp_ms", -1) + 1
    detector._extraction_timestamp_ms = timestamp_ms
    result = detector.detect_for_video(mp_image, timestamp_ms)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def run_face_mesh_demo():
    """Run the live webcam demo for face landmark detection and overlays."""
    # Keep the display tied to the most recently captured frame.  Some camera
    # backends otherwise queue several frames, which makes the iris markers look
    # as though they are trailing the user's eyes.
    capture_started_at = time.perf_counter()
    timestamp_ms = -1
    for frame in read_webcam_frames():
        if legacy_face_mesh:
            # Convert OpenCV's BGR frame to the RGB format expected by MediaPipe.
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                # Draw the complete mesh first, then add the derived overlays.
                for lm in face_landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                draw_gaze_overlay(frame, face_landmarks)
                draw_face_pose(frame, face_landmarks)
        else:
            # The Tasks API uses an explicit MediaPipe image wrapper.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
            # VIDEO mode uses this value for temporal tracking.  It must reflect
            # elapsed time rather than loop iterations; a 1 ms increment per
            # webcam frame makes MediaPipe's tracker use the wrong timing.
            elapsed_ms = int((time.perf_counter() - capture_started_at) * 1000)
            timestamp_ms = max(timestamp_ms + 1, elapsed_ms)
            result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            for face in result.face_landmarks:
                # Each detected face is drawn and analyzed independently.
                for lm in face:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                draw_gaze_overlay(frame, face)
                draw_face_pose(frame, face)

        # Show the annotated frame until the user presses q.
        cv2.imshow("Face Mesh test", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_face_mesh_demo()
