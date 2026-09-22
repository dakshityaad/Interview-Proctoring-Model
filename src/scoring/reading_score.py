"""Persistent probability scoring for the live reading detector."""


class ReadingScore:
	"""Apply hysteresis and consecutive-window rules to reading probabilities."""

	def __init__(
		self,
		positive_threshold=0.70,
		negative_threshold=0.40,
		consecutive_positive_windows=2,
		consecutive_negative_windows=2,
	):
		if not 0 <= negative_threshold < positive_threshold <= 1:
			raise ValueError("thresholds must satisfy 0 <= negative < positive <= 1")
		if consecutive_positive_windows <= 0 or consecutive_negative_windows <= 0:
			raise ValueError("consecutive window counts must be positive")
		self.positive_threshold = positive_threshold
		self.negative_threshold = negative_threshold
		self.consecutive_positive_windows = consecutive_positive_windows
		self.consecutive_negative_windows = consecutive_negative_windows
		self.is_reading = False
		self._positive_count = 0
		self._negative_count = 0

	def update(self, reading_probability):
		"""Consume one probability and return the current reading state."""
		if not 0 <= reading_probability <= 1:
			raise ValueError("reading_probability must be between 0 and 1")

		if not self.is_reading:
			self._positive_count = self._positive_count + 1 if reading_probability >= self.positive_threshold else 0
			if self._positive_count >= self.consecutive_positive_windows:
				self.is_reading = True
				self._negative_count = 0
		else:
			self._negative_count = self._negative_count + 1 if reading_probability <= self.negative_threshold else 0
			if self._negative_count >= self.consecutive_negative_windows:
				self.is_reading = False
				self._positive_count = 0
		return self.is_reading

	def reset(self):
		"""Clear accumulated evidence for a new video/session."""
		self.is_reading = False
		self._positive_count = 0
		self._negative_count = 0