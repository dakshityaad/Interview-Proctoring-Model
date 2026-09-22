"""Train the five-second binary reading detector."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

import yaml

from src.model.cnn_model import build_model
from src.model.dataset import split_by_subject
from src.model.train import save_model, train_model
from src.windowing.make_windows import build_reading_dataset


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--features-csv", type=Path, default=ROOT / "data" / "features.csv")
	parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "reading_cnn_classifier.pt")
	parser.add_argument("--test-subject", default=None)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	args = parser.parse_args()

	with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	reading = config["reading_detection"]
	fps = float(reading["sampling_fps"])
	window_size = round(float(reading["window_seconds"]) * fps)
	stride = max(1, round(float(reading["stride_seconds"]) * fps))
	dataset = build_reading_dataset(args.features_csv, window_size, stride, fps)
	subjects = sorted({item[2] for item in dataset})
	test_subject = args.test_subject or subjects[-1]
	train_data, val_data = split_by_subject(dataset, test_subject)
	model = build_model(input_features=10, num_classes=2)

	# Inverse-frequency weights keep the smaller reading class visible to the loss.
	counts = [sum(item[1] == label for item in train_data) for label in (0, 1)]
	total = sum(counts)
	class_weights = [total / (2 * count) if count else 1.0 for count in counts]
	history = train_model(
		model,
		train_data,
		val_data,
		args.epochs,
		learning_rate=args.learning_rate,
		batch_size=args.batch_size,
		class_weights=class_weights,
	)
	save_model(model, args.model_path)
	print(f"Window: {window_size} frames ({reading['window_seconds']} seconds at {fps:g} FPS)")
	print(f"Held-out subject: {test_subject}")
	print(f"Train windows: {len(train_data)}; validation windows: {len(val_data)}")
	print(f"Class weights: {class_weights}")
	print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2%}")
	print(f"Saved model: {args.model_path}")


if __name__ == "__main__":
	main()