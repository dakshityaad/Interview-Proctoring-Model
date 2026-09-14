import cv2
import numpy as np
import time

try:
    import mediapipe as mp
except ImportError as exc:
    raise RuntimeError(
        "MediaPipe is not installed. Run:\n"
        "py -3.12 -m venv .venv312\n"
        ".\\.venv312\\Scripts\\python.exe -m pip install -r requirements.txt"
    ) from exc


# 3D reference points corresponding to the six MediaPipe landmarks below.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
POSE_LANDMARK_IDS = [4, 152, 263, 33, 287, 61]

# Iris + eye-corner landmarks for gaze-ratio estimation
RIGHT_IRIS, LEFT_IRIS = 468, 473
RIGHT_EYE = {"inner": 133, "outer": 33, "top": 159, "bottom": 145}
LEFT_EYE  = {"inner": 362, "outer": 263, "top": 386, "bottom": 374}


def get_gaze_ratios(face_landmarks, img_w, img_h):
    """Return (horizontal_ratio, vertical_ratio), each in [0, 1].

    0.5 = iris centered in the eye socket (looking straight).
    Values toward 0 or 1 mean the iris is pushed to one side/up/down.
    """
    def eye_ratios(iris_id, corners):
        ix = face_landmarks[iris_id].x * img_w
        iy = face_landmarks[iris_id].y * img_h
        cx = {k: face_landmarks[v].x * img_w for k, v in corners.items()}
        cy = {k: face_landmarks[v].y * img_h for k, v in corners.items()}

        horizontal_span = cx["outer"] - cx["inner"]
        vertical_span = cy["bottom"] - cy["top"]
        if abs(horizontal_span) < 1e-6 or abs(vertical_span) < 1e-6:
            return 0.5, 0.5
        horiz = (ix - cx["inner"]) / horizontal_span
        vert = (iy - cy["top"]) / vertical_span
        return horiz, vert

    r = eye_ratios(RIGHT_IRIS, RIGHT_EYE)
    l = eye_ratios(LEFT_IRIS, LEFT_EYE)
    return (r[0] + l[0]) / 2, (r[1] + l[1]) / 2


def draw_gaze_overlay(frame, face_landmarks):
    """Draw iris markers and a compact gaze-position indicator."""
    height, width = frame.shape[:2]
    if len(face_landmarks) <= LEFT_IRIS:
        return

    gx, gy = get_gaze_ratios(face_landmarks, width, height)
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


def get_head_pose(face_landmarks, img_w, img_h):
    """Return pose angles and solvePnP data for normalized face landmarks."""
    img_points = np.array([
        [face_landmarks[i].x * img_w, face_landmarks[i].y * img_h]
        for i in POSE_LANDMARK_IDS
    ], dtype=np.float64)
    focal = float(img_w)
    cam_matrix = np.array([
        [focal, 0, img_w / 2],
        [0, focal, img_h / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, img_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = angles
    return pitch, yaw, roll, rvec, tvec, cam_matrix, dist_coeffs


def draw_face_pose(frame, face_landmarks):
    """Draw the 3D head axes and live pitch/yaw/roll overlay."""
    height, width = frame.shape[:2]
    pose = get_head_pose(face_landmarks, width, height)
    if pose is None:
        return

    pitch, yaw, roll, rvec, tvec, cam_matrix, dist_coeffs = pose
    nose = (int(face_landmarks[4].x * width), int(face_landmarks[4].y * height))
    axis_points, _ = cv2.projectPoints(
        np.array([(100.0, 0, 0), (0, 100.0, 0), (0, 0, 100.0)]),
        rvec, tvec, cam_matrix, dist_coeffs,
    )
    projected = axis_points.reshape(3, 2).astype(int)
    cv2.line(frame, nose, tuple(projected[0]), (0, 0, 255), 2)
    cv2.line(frame, nose, tuple(projected[1]), (0, 255, 0), 2)
    cv2.line(frame, nose, tuple(projected[2]), (255, 0, 0), 2)
    cv2.putText(
        frame,
        f"P {pitch:.1f} Y {yaw:.1f} R {roll:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


legacy_face_mesh = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")

if legacy_face_mesh:
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
else:
    try:
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

# Keep the display tied to the most recently captured frame.  Some camera
# backends otherwise queue several frames, which makes the iris markers look
# as though they are trailing the user's eyes.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
capture_started_at = time.perf_counter()
timestamp_ms = -1
while True:
    ok, frame = cap.read()
    if not ok:
        break

    if legacy_face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            for lm in face_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            draw_gaze_overlay(frame, face_landmarks)
            draw_face_pose(frame, face_landmarks)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        # VIDEO mode uses this value for temporal tracking.  It must reflect
        # elapsed time rather than loop iterations; a 1 ms increment per
        # webcam frame makes MediaPipe's tracker use the wrong timing.
        elapsed_ms = int((time.perf_counter() - capture_started_at) * 1000)
        timestamp_ms = max(timestamp_ms + 1, elapsed_ms)
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        for face in result.face_landmarks:
            for lm in face:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            draw_gaze_overlay(frame, face)
            draw_face_pose(frame, face)

    cv2.imshow("Face Mesh test", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
