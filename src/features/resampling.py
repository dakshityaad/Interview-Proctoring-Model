"""Timestamp-aware resampling of frame-level face features."""

import numpy as np


def resample_features(timestamps, values, target_fps, max_gap_seconds=None):
	"""Resample onto a fixed timeline without bridging long detection gaps."""
	timestamps = np.asarray(timestamps, dtype=np.float64)
	values = np.asarray(values, dtype=np.float32)
	if target_fps <= 0:
		raise ValueError("target_fps must be positive")
	if timestamps.ndim != 1 or values.ndim != 2 or len(timestamps) != len(values):
		raise ValueError("timestamps must be 1D and match the number of feature rows")
	if len(timestamps) < 2:
		return np.empty((0, values.shape[1]), dtype=np.float32)
	if np.any(np.diff(timestamps) <= 0):
		raise ValueError("timestamps must be strictly increasing within a recording")

	if max_gap_seconds is None:
		max_gap_seconds = 2.0 / float(target_fps)
	if max_gap_seconds <= 0:
		raise ValueError("max_gap_seconds must be positive")

	start_index = int(np.ceil(timestamps[0] * target_fps))
	end_index = int(np.floor(timestamps[-1] * target_fps))
	if end_index < start_index:
		return np.empty((0, values.shape[1]), dtype=np.float32)
	target_times = np.arange(start_index, end_index + 1, dtype=np.float64) / target_fps
	resampled = np.full((len(target_times), values.shape[1]), np.nan, dtype=np.float32)

	for source_index in range(len(timestamps) - 1):
		left_time = timestamps[source_index]
		right_time = timestamps[source_index + 1]
		if right_time - left_time > max_gap_seconds:
			continue
		mask = (target_times >= left_time) & (target_times <= right_time)
		if not np.any(mask):
			continue
		for column in range(values.shape[1]):
			resampled[mask, column] = np.interp(
				target_times[mask],
				[left_time, right_time],
				values[source_index:source_index + 2, column],
			)

	# Include exact source observations and never extrapolate beyond the recording.
	exact = np.searchsorted(timestamps, target_times)
	for index, source_index in enumerate(exact):
		if source_index < len(timestamps) and np.isclose(target_times[index], timestamps[source_index]):
			resampled[index] = values[source_index]

	return resampled
