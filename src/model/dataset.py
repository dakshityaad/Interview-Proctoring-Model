"""Prepare subject-aware window datasets for the PyTorch models.

The idea in plain English:
	1. Keep every window from one subject together.
	2. Hold out one complete subject for an honest test set.
	3. Convert the remaining Python/numpy values into PyTorch tensors.

The preferred dataset item is ``(window, class_label, subject_id)``. A mapping
with ``window``, ``class_label`` (or ``label``), and ``subject_id`` keys is also
accepted. Subject metadata is required because a two-item ``(window, label)``
record cannot provide a safe subject-level split.
"""

import torch


def _unpack_item(item):
	"""Return ``(window, label, subject_id)`` from a supported item shape."""
	if isinstance(item, dict):
		try:
			label = item["class_label"] if "class_label" in item else item["label"]
			return item["window"], label, item["subject_id"]
		except KeyError as exc:
			raise ValueError(
				"Dataset mappings need window, class_label/label, and subject_id keys"
			) from exc

	if len(item) != 3:
		raise ValueError(
			"Subject-level splitting requires dataset items shaped "
			"(window, class_label, subject_id)"
		)
	return item[0], item[1], item[2]


def split_by_subject(dataset, test_subject_id):
	"""Split windows while holding out every window from one subject.

	Args:
		dataset: Iterable of subject-aware dataset items.
		test_subject_id: Subject identifier reserved for testing.

	Returns:
		``(train_dataset, test_dataset)`` in the original item format.
	"""
	train_dataset = []
	test_dataset = []

	for item in dataset:
		_, _, subject_id = _unpack_item(item)
		if subject_id == test_subject_id:
			test_dataset.append(item)
		else:
			train_dataset.append(item)

	if not test_dataset:
		raise ValueError(f"No dataset items found for test subject {test_subject_id!r}")

	return train_dataset, test_dataset


def to_tensors(dataset):
	"""Convert windows and labels into ``(features, labels)`` PyTorch tensors.

	The feature tensor has shape ``(samples, frames, features)`` and uses
	``float32`` values. Labels have shape ``(samples,)`` and use ``int64``,
	which is the dtype expected by PyTorch cross-entropy loss.
	"""
	items = list(dataset)
	if not items:
		raise ValueError("Cannot convert an empty dataset to tensors")

	windows = []
	labels = []
	for item in items:
		window, label, _ = _unpack_item(item)
		windows.append(window)
		labels.append(label)

	features = torch.as_tensor(windows, dtype=torch.float32)
	labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
	return features, labels_tensor
