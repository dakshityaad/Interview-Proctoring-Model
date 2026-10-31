import unittest

import numpy as np
import torch

from src.features.motion import add_motion_features
from src.features.resampling import resample_features
from src.model.ann_model import build_model as build_ann_model
from src.model.dnn_model import build_model as build_dnn_model
from src.model.cnn_lstm_model import build_model as build_cnn_lstm_model
from src.model.dnn_lstm_model import build_model as build_dnn_lstm_model
from src.scoring.reading_score import ReadingScore
from src.windowing.make_windows import build_reading_dataset, fill_short_missing_gaps


class ModelArchitectureTests(unittest.TestCase):
	def test_ann_flattens_and_classifies_reading_windows(self):
		model = build_ann_model(input_features=10, frames=38, num_classes=2)
		inputs = torch.zeros((4, 38, 10), dtype=torch.float32)

		outputs = model(inputs)

		self.assertEqual(outputs.shape, (4, 2))
		self.assertTrue(torch.allclose(outputs.sum(dim=1), torch.ones(4)))

	def test_ann_rejects_wrong_window_shape(self):
		model = build_ann_model(input_features=10, frames=38, num_classes=2)

		with self.assertRaises(ValueError):
			model(torch.zeros((1, 60, 10)))

	def test_dnn_uses_deep_hidden_layer_stack(self):
		model = build_dnn_model(input_features=10, frames=38, num_classes=2)
		inputs = torch.zeros((4, 38, 10), dtype=torch.float32)

		outputs = model(inputs)

		self.assertEqual(outputs.shape, (4, 2))
		self.assertTrue(torch.allclose(outputs.sum(dim=1), torch.ones(4)))
		hidden_sizes = [128, 64, 32, 16]
		linear_sizes = [layer.out_features for layer in model.classifier if isinstance(layer, torch.nn.Linear)][:-1]
		self.assertEqual(linear_sizes, hidden_sizes)

	def test_dnn_rejects_wrong_window_shape(self):
		model = build_dnn_model(input_features=10, frames=38, num_classes=2)

		with self.assertRaises(ValueError):
			model(torch.zeros((1, 60, 10)))

	def test_cnn_lstm_classifies_temporal_windows(self):
		model = build_cnn_lstm_model(input_features=3, num_classes=3)
		inputs = torch.zeros((4, 38, 3), dtype=torch.float32)

		outputs = model(inputs)

		self.assertEqual(outputs.shape, (4, 3))
		self.assertTrue(torch.allclose(outputs.sum(dim=1), torch.ones(4)))

	def test_dnn_lstm_classifies_temporal_windows(self):
		model = build_dnn_lstm_model(input_features=3, frames=38, num_classes=3)
		inputs = torch.zeros((4, 38, 3), dtype=torch.float32)

		outputs = model(inputs)

		self.assertEqual(outputs.shape, (4, 3))
		self.assertTrue(torch.allclose(outputs.sum(dim=1), torch.ones(4)))


class ReadingPipelineTests(unittest.TestCase):
	def test_motion_features_append_seven_channels(self):
		sequence = np.zeros((38, 3), dtype=np.float32)
		sequence[:, 1] = np.arange(38, dtype=np.float32)
		features = add_motion_features(sequence, sampling_fps=7.5)

		self.assertEqual(features.shape, (38, 10))
		self.assertTrue(np.all(features[:, -1] >= 0))

	def test_timestamp_resampling_normalizes_native_frame_rates(self):
		timestamps = np.arange(0, 5, 1 / 30, dtype=np.float64)
		values = np.column_stack([timestamps, timestamps * 2, timestamps * 3])

		resampled = resample_features(timestamps, values, target_fps=7.5)

		self.assertEqual(resampled.shape, (38, 3))
		self.assertAlmostEqual(float(resampled[-1, 0]), 4.9333, places=3)

	def test_reading_dataset_requires_timestamped_features_for_resampling(self):
		with self.assertRaisesRegex(ValueError, "timestamp_seconds"):
			build_reading_dataset("data/features.csv")

	def test_long_resampling_gap_remains_missing(self):
		timestamps = np.array([0.0, 0.1, 2.0, 2.1], dtype=np.float64)
		values = np.column_stack([timestamps, timestamps, timestamps])

		resampled = resample_features(
			timestamps, values, target_fps=7.5, max_gap_seconds=0.4,
		)

		self.assertTrue(np.isnan(resampled).any())

	def test_short_internal_gap_is_filled(self):
		sequence = np.ones((38, 3), dtype=np.float32)
		sequence[10:12] = np.nan

		filled = fill_short_missing_gaps(sequence, fps=7.5, max_gap_seconds=0.4)

		self.assertTrue(np.isfinite(filled).all())

	def test_reading_score_requires_persistent_evidence(self):
		score = ReadingScore()

		self.assertFalse(score.update(0.9))
		self.assertTrue(score.update(0.9))
		self.assertTrue(score.update(0.2))
		self.assertFalse(score.update(0.2))
