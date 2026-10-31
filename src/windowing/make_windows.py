"""Create labeled temporal windows from the extracted feature CSV."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.motion import add_motion_features
from src.features.resampling import resample_features


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


def fill_short_missing_gaps(sequence, fps, max_gap_seconds):
	"""Fill only short internal gaps; retain long and boundary gaps as NaN."""
	values = np.asarray(sequence, dtype=np.float32).copy()
	max_frames = max(1, int(np.ceil(max_gap_seconds * fps)))

	for column in range(values.shape[1]):
		column_values = values[:, column]
		column_missing = ~np.isfinite(column_values)
		for run_start, run_end in _missing_runs(column_missing):
			if run_start == 0 or run_end == len(column_values):
				continue
			if run_end - run_start <= max_frames:
				column_values[run_start:run_end] = np.linspace(
					column_values[run_start - 1],
					column_values[run_end],
					run_end - run_start + 2,
				)[1:-1]
	return values


def _missing_runs(mask):
	"""Yield [start, end) ranges for contiguous True values."""
	starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
	ends = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
	return zip(starts, ends)


def build_dataset(
	csv_path,
	window_size=60,
	stride=10,
	include_motion=False,
	sampling_fps=15.0,
	binary_reading=False,
	resample_fps=None,
	max_gap_seconds=None,
	max_missing_fraction=0.10,
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
		if resample_fps is not None:
			if "timestamp_seconds" not in group.columns:
				raise ValueError(
					"Timestamp-based resampling requires a timestamp_seconds column. "
					"Re-extract features from the raw videos with scripts/run_extraction.py."
				)
			sequence = resample_features(
				group["timestamp_seconds"].to_numpy(dtype="float64"),
				sequence,
				resample_fps,
				max_gap_seconds=max_gap_seconds,
			)
			gap_seconds = max_gap_seconds
			if gap_seconds is None:
				gap_seconds = 2.0 / float(resample_fps)
			sequence = fill_short_missing_gaps(sequence, resample_fps, gap_seconds)
		if include_motion:
			sequence = add_motion_features(sequence, sampling_fps=sampling_fps)
		label = int(class_label == 2) if binary_reading else int(class_label)
		for window in slice_into_windows(sequence, window_size, stride):
			missing_fraction = np.count_nonzero(~np.isfinite(window)) / window.size
			if missing_fraction > max_missing_fraction:
				continue
			if not np.all(np.isfinite(window)):
				continue
			dataset.append((window, label, subject_id))

	if not dataset:
		raise ValueError(
		f"No windows could be created with window_size={window_size} "
		f"and stride={stride}"
	)
	return dataset


def build_reading_dataset(
	csv_path,
	window_size=38,
	stride=4,
	sampling_fps=7.5,
	max_gap_seconds=None,
	max_missing_fraction=0.10,
):
	"""Build five-second binary windows where label 1 means reading."""
	return build_dataset(
		csv_path,
		window_size=window_size,
		stride=stride,
		include_motion=True,
		sampling_fps=sampling_fps,
		binary_reading=True,
		resample_fps=sampling_fps,
		max_gap_seconds=max_gap_seconds,
		max_missing_fraction=max_missing_fraction,
	)
