"""Training and checkpoint helpers for the classifier models."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.dataset import to_tensors


def _loader(data, batch_size, shuffle):
	features, labels = to_tensors(data) if not isinstance(data, tuple) else data
	return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_data, val_data, epochs, learning_rate=1e-3, batch_size=64):
	"""Train a model and return per-epoch loss and accuracy history."""
	if epochs <= 0:
		raise ValueError("epochs must be positive")
	device = next(model.parameters()).device
	train_loader = _loader(train_data, batch_size, shuffle=True)
	val_loader = _loader(val_data, batch_size, shuffle=False)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.NLLLoss()
	history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

	for _ in range(epochs):
		model.train()
		total_loss = 0.0
		for features, labels in train_loader:
			features, labels = features.to(device), labels.to(device)
			optimizer.zero_grad()
			probabilities = model(features)
			loss = criterion(torch.log(probabilities.clamp_min(1e-8)), labels)
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * labels.size(0)

		model.eval()
		val_loss = 0.0
		correct = 0
		total = 0
		with torch.no_grad():
			for features, labels in val_loader:
				features, labels = features.to(device), labels.to(device)
				probabilities = model(features)
				val_loss += criterion(torch.log(probabilities.clamp_min(1e-8)), labels).item() * labels.size(0)
				correct += (probabilities.argmax(dim=1) == labels).sum().item()
				total += labels.size(0)

		history["train_loss"].append(total_loss / len(train_loader.dataset))
		history["val_loss"].append(val_loss / len(val_loader.dataset))
		history["val_accuracy"].append(correct / total)

	return history


def save_model(model, path):
	"""Save model weights to ``path`` and create its parent directory."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), path)
