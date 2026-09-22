"""Streaming five-second reading detector for webcam integrations."""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.features.motion import add_motion_features
from src.model.cnn_model import build_model
from src.scoring.reading_score import ReadingScore


class ReadingDetector:
	"""Accumulate frame features and classify the latest five-second window."""

	def __init__(
		self,
		model_path=None,
		sampling_fps=15.0,
		window_seconds=5.0,
		positive_threshold=0.70,
		negative_threshold=0.40,
		consecutive_positive_windows=2,
		consecutive_negative_windows=2,
		device="cpu",
	):
		if sampling_fps <= 0 or window_seconds <= 0:
			raise ValueError("sampling_fps and window_seconds must be positive")
		self.device = torch.device(device)
		self.window_size = round(sampling_fps * window_seconds)
		self.frames = deque(maxlen=self.window_size)
		self.model = build_model(input_features=10, num_classes=2).to(self.device)
		if model_path is not None:
			state_dict = torch.load(Path(model_path), map_location=self.device, weights_only=True)
			self.model.load_state_dict(state_dict)
		self.model.eval()
		self.sampling_fps = sampling_fps
		self.score = ReadingScore(
			positive_threshold=positive_threshold,
			negative_threshold=negative_threshold,
			consecutive_positive_windows=consecutive_positive_windows,
			consecutive_negative_windows=consecutive_negative_windows,
		)

	def update(self, pitch, yaw, gaze_ratio):
		"""Add one frame and return a result once the five-second buffer is full."""
		self.frames.append((pitch, yaw, gaze_ratio))
		if len(self.frames) < self.window_size:
			return None

		sequence = add_motion_features(
			np.asarray(self.frames, dtype=np.float32),
			sampling_fps=self.sampling_fps,
		)
		inputs = torch.as_tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
		with torch.no_grad():
			probabilities = self.model(inputs)[0]
		reading_probability = float(probabilities[1].item())
		return {
			"reading_probability": reading_probability,
			"is_reading": self.score.update(reading_probability),
		}

	def reset(self):
		"""Clear buffered frames and alert history for a new stream."""
		self.frames.clear()
		self.score.reset()