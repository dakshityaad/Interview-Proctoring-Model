"""A compact 1D CNN for temporal face-feature windows."""

import torch
from torch import nn


class CNNClassifier(nn.Module):
	"""Classify sequences shaped ``(batch, frames, features)``."""

	def __init__(self, input_features=3, num_classes=3):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv1d(input_features, 32, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(32, 64, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool1d(1),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, num_classes),
		)

	def forward(self, inputs):
		if inputs.ndim != 3:
			raise ValueError("CNN inputs must have shape (batch, frames, features)")
		# Conv1d expects channels before the temporal dimension.
		logits = self.classifier(self.features(inputs.transpose(1, 2)))
		return torch.softmax(logits, dim=1)


def build_model(input_features=3, num_classes=3):
	"""Build the default CNN classifier."""
	return CNNClassifier(input_features=input_features, num_classes=num_classes)
