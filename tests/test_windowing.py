import unittest

import numpy as np

from src.features.motion import add_motion_features
from src.scoring.reading_score import ReadingScore
from src.windowing.make_windows import build_reading_dataset


class ReadingPipelineTests(unittest.TestCase):
	def test_motion_features_append_seven_channels(self):
		sequence = np.zeros((75, 3), dtype=np.float32)
		sequence[:, 1] = np.arange(75, dtype=np.float32)
		features = add_motion_features(sequence, sampling_fps=15)

		self.assertEqual(features.shape, (75, 10))
		self.assertTrue(np.all(features[:, -1] >= 0))

	def test_reading_windows_are_five_seconds_at_configured_rate(self):
		dataset = build_reading_dataset("data/features.csv", window_size=75, stride=8, sampling_fps=15)

		self.assertTrue(dataset)
		self.assertEqual(dataset[0][0].shape, (75, 10))
		self.assertEqual({item[1] for item in dataset}, {0, 1})

	def test_reading_score_requires_persistent_evidence(self):
		score = ReadingScore()

		self.assertFalse(score.update(0.9))
		self.assertTrue(score.update(0.9))
		self.assertTrue(score.update(0.2))
		self.assertFalse(score.update(0.2))
