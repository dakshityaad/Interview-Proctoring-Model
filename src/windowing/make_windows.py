"""Create labeled temporal windows from the extracted feature CSV."""

from pathlib import Path

import pandas as pd

from src.features.motion import add_motion_features


FEATURE_COLUMNS = ["pitch", "yaw", "gaze_ratio"]


def load_features(csv_path):
	"""Load and validate frame-level features from ``csv_path``."""
	frame_data = pd.read_csv(Path(csv_path))
	required_columns = {"subject_id", "class_label", "frame_number", *FEATURE_COLUMNS}
	missing_columns = required_columns.difference(frame_data.columns)
	if missing_columns:
		raise ValueError(f"Feature CSV is missing columns: {sorted(missing_columns)}")
	if frame_data.empty:
		raise ValueError("Feature CSV contains no rows")
	return frame_data


def slice_into_windows(frame_sequence, window_size, stride):
	"""Return overlapping windows from a sequence of frame feature vectors."""
	if window_size <= 0 or stride <= 0:
		raise ValueError("window_size and stride must be positive")
	return [
		frame_sequence[start:start + window_size]
		for start in range(0, len(frame_sequence) - window_size + 1, stride)
	]


def build_dataset(
	csv_path,
	window_size=60,
	stride=10,
	include_motion=False,
	sampling_fps=15.0,
	binary_reading=False,
):
	"""Build subject/class-safe windows for the existing or reading model."""
	frame_data = load_features(csv_path)
	dataset = []

	# Grouping by subject and class prevents a window from crossing a recording boundary.
	for (subject_id, class_label), group in frame_data.groupby(
		["subject_id", "class_label"], sort=True
	):
		group = group.sort_values("frame_number")
		sequence = group[FEATURE_COLUMNS].to_numpy(dtype="float32")
		if include_motion:
			sequence = add_motion_features(sequence, sampling_fps=sampling_fps)
		label = int(class_label == 2) if binary_reading else int(class_label)
		for window in slice_into_windows(sequence, window_size, stride):
			dataset.append((window, label, subject_id))

	if not dataset:
		raise ValueError(
		f"No windows could be created with window_size={window_size} "
		f"and stride={stride}"
	)
	return dataset


def build_reading_dataset(csv_path, window_size=75, stride=8, sampling_fps=15.0):
	"""Build five-second binary windows where label 1 means reading."""
	return build_dataset(
		csv_path,
		window_size=window_size,
		stride=stride,
		include_motion=True,
		sampling_fps=sampling_fps,
		binary_reading=True,
	)
