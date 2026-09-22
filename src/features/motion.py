"""Temporal motion features for the reading detector."""

import numpy as np


BASE_FEATURE_COUNT = 3
MOTION_FEATURE_NAMES = [
	"pitch_velocity",
	"yaw_velocity",
	"gaze_velocity",
	"pitch_acceleration",
	"yaw_acceleration",
	"gaze_acceleration",
	"horizontal_motion",
]


def add_motion_features(sequence, sampling_fps=15.0):
	"""Append first/second derivatives and horizontal motion to a sequence."""
	values = np.asarray(sequence, dtype=np.float32)
	if values.ndim != 2 or values.shape[1] != BASE_FEATURE_COUNT:
		raise ValueError("sequence must have shape (frames, 3)")
	if sampling_fps <= 0:
		raise ValueError("sampling_fps must be positive")
	if len(values) == 0:
		return np.empty((0, BASE_FEATURE_COUNT + len(MOTION_FEATURE_NAMES)), dtype=np.float32)

	step = 1.0 / float(sampling_fps)
	velocity = np.gradient(values, step, axis=0)
	acceleration = np.gradient(velocity, step, axis=0)
	horizontal_motion = np.abs(velocity[:, 1]) + np.abs(velocity[:, 2])
	motion = np.column_stack(
		[
			velocity[:, 0],
			velocity[:, 1],
			velocity[:, 2],
			acceleration[:, 0],
			acceleration[:, 1],
			acceleration[:, 2],
			horizontal_motion,
		]
	)
	return np.concatenate([values, motion.astype(np.float32)], axis=1)