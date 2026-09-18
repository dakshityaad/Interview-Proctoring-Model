"""Train the CNN classifier from the extracted feature CSV."""

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
from src.windowing.make_windows import build_dataset


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--features-csv", type=Path, default=ROOT / "data" / "features.csv")
	parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "cnn_classifier.pt")
	parser.add_argument("--test-subject", default=None)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	args = parser.parse_args()

	with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	dataset = build_dataset(
		args.features_csv,
		window_size=config["windowing"]["window_size"],
		stride=config["windowing"]["stride"],
	)
	subjects = sorted({item[2] for item in dataset})
	test_subject = args.test_subject or subjects[-1]
	train_data, val_data = split_by_subject(dataset, test_subject)
	model = build_model(num_classes=len(config["classes"]))
	history = train_model(
		model, train_data, val_data, args.epochs,
		learning_rate=args.learning_rate, batch_size=args.batch_size,
	)
	save_model(model, args.model_path)
	print(f"Held-out subject: {test_subject}")
	print(f"Train windows: {len(train_data)}; validation windows: {len(val_data)}")
	print(f"Initial loss: {history['train_loss'][0]:.4f}")
	print(f"Final loss: {history['train_loss'][-1]:.4f}")
	print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2%}")
	print(f"Saved model: {args.model_path}")


if __name__ == "__main__":
	main()
