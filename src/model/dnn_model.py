"""Deep fully connected network for flattened feature windows."""

import torch
from torch import nn


class DNNClassifier(nn.Module):
	"""A four-hidden-layer fully connected comparison model."""

	def __init__(self, input_features=3, frames=60, num_classes=3, dropout=0.2):
		if input_features <= 0 or frames <= 0 or num_classes <= 0:
			raise ValueError("input_features, frames, and num_classes must be positive")
		if not 0 <= dropout < 1:
			raise ValueError("dropout must be in the range [0, 1)")

		super().__init__()
		self.input_features = input_features
		self.frames = frames
		self.input_size = input_features * frames
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.input_size, 128),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(32, 16),
			nn.ReLU(),
			nn.Linear(16, num_classes),
		)

	def forward(self, inputs):
		if inputs.ndim != 3:
			raise ValueError("DNN inputs must have shape (batch, frames, features)")
		if tuple(inputs.shape[1:]) != (self.frames, self.input_features):
			raise ValueError(
				f"Expected window shape {(self.frames, self.input_features)}, "
				f"received {tuple(inputs.shape[1:])}"
			)
		return torch.softmax(self.classifier(inputs), dim=1)


def build_model(input_features=3, frames=60, num_classes=3, dropout=0.2):
	"""Build the project DNN with 128, 64, 32, and 16-unit hidden layers."""
	return DNNClassifier(
		input_features=input_features,
		frames=frames,
		num_classes=num_classes,
		dropout=dropout,
	)
