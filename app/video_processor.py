"""Live webcam feature extraction and CNN+LSTM inference."""

from collections import deque
import threading
import time

import numpy as np
import torch

from src.landmarks import face_mesh, gaze, head_pose
from src.model.cnn_lstm_model import build_model


CLASS_NAMES = ("Talking", "Looking down", "Reading")


class LiveInferenceProcessor:
    """Convert webcam frames to 38-feature windows and run the selected model."""

    def __init__(self, model_path, window_size=38, sampling_fps=7.5, max_missing_fraction=0.10):
        self.window_size = window_size
        self.sample_interval = 1.0 / sampling_fps
        self.max_missing_fraction = max_missing_fraction
        self.features = deque(maxlen=window_size)
        self.lock = threading.Lock()
        self.detector = face_mesh.init_face_mesh()
        self.model = build_model(input_features=3, num_classes=3, dropout=0.2)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.last_sample_time = None
        self.last_prediction = None
        self.prediction_history = deque(maxlen=300)

    def process_frame(self, frame, timestamp=None):
        """Sample one webcam frame at 7.5 FPS and optionally return a prediction."""
        now = time.monotonic() if timestamp is None else timestamp
        if self.last_sample_time is not None and now - self.last_sample_time < self.sample_interval:
            return self.snapshot()
        self.last_sample_time = now

        frame_height, frame_width = frame.shape[:2]
        landmarks = face_mesh.get_landmarks(self.detector, frame)
        face_visible = landmarks is not None
        if face_visible:
            pitch, yaw = head_pose.get_head_pose(landmarks, frame_width, frame_height)
            horizontal_gaze = gaze.get_gaze_ratio(landmarks, frame_width, frame_height)[0]
            if pitch is None or yaw is None or not np.isfinite(horizontal_gaze):
                features = [np.nan, np.nan, np.nan]
            else:
                features = [pitch, yaw, horizontal_gaze]
        else:
            features = [np.nan, np.nan, np.nan]

        with self.lock:
            self.features.append(np.asarray(features, dtype=np.float32))
            result = self._predict_locked(face_visible)
        if result is not None:
            self.last_prediction = result
            self.prediction_history.append((time.monotonic(), result["class_name"], result["confidence"]))
        return {
            "face_visible": face_visible,
            "prediction": result or self.last_prediction,
            "buffer_size": len(self.features),
        }

    def _predict_locked(self, face_visible):
        del face_visible
        if len(self.features) < self.window_size:
            return None
        sequence = np.stack(list(self.features)).astype(np.float32)
        missing_fraction = np.count_nonzero(~np.isfinite(sequence)) / sequence.size
        if missing_fraction > self.max_missing_fraction:
            return None
        if not np.all(np.isfinite(sequence)):
            # Training windows were filtered to fully finite sequences after
            # filling short gaps. Live inference deliberately does not interpolate.
            return None
        tensor = torch.from_numpy(sequence).unsqueeze(0)
        with torch.no_grad():
            probabilities = self.model(tensor).cpu().numpy()[0]
        predicted_index = int(np.argmax(probabilities))
        return {
            "class_index": predicted_index,
            "class_name": CLASS_NAMES[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {name: float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)},
        }

    def snapshot(self):
        with self.lock:
            prediction = self.last_prediction
            buffer_size = len(self.features)
        return {"prediction": prediction, "buffer_size": buffer_size, "face_visible": None}

    def history_snapshot(self):
        with self.lock:
            return list(self.prediction_history)

    def reset(self):
        with self.lock:
            self.features.clear()
            self.last_prediction = None
            self.prediction_history.clear()
        self.last_sample_time = None

