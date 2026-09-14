import cv2

try:
    import mediapipe as mp
except ImportError as exc:
    raise RuntimeError(
        "MediaPipe is not installed. Run:\n"
        "py -3.12 -m venv .venv312\n"
        ".\\.venv312\\Scripts\\python.exe -m pip install -r requirements.txt"
    ) from exc

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

cap = cv2.VideoCapture(0)
timestamp_ms = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    if legacy_face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks[0].landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        timestamp_ms += 1
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        for face in result.face_landmarks:
            for lm in face:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Face Mesh test", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()