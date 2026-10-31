"""Train the CNN classifier from the extracted feature CSV."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

import yaml

from src.model.ann_model import build_model as build_ann_model
from src.model.cnn_model import build_model as build_cnn_model
from src.model.dataset import split_by_subject
from src.model.dnn_model import build_model as build_dnn_model
from src.model.cnn_lstm_model import build_model as build_cnn_lstm_model
from src.model.dnn_lstm_model import build_model as build_dnn_lstm_model
from src.model.train import save_model, train_model
from src.windowing.make_windows import build_dataset


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--features-csv", type=Path, default=ROOT / "data" / "features.csv")
	parser.add_argument("--model-path", type=Path, default=None)
	parser.add_argument("--test-subject", default=None)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--model-type", choices=("cnn", "ann", "dnn", "cnn_lstm", "dnn_lstm"), default="cnn")
	args = parser.parse_args()
	if args.model_path is None:
		args.model_path = ROOT / "models" / f"{args.model_type}_classifier.pt"

	with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	window_size = config["windowing"]["window_size"]
	dataset = build_dataset(
		args.features_csv,
		window_size=window_size,
		stride=config["windowing"]["stride"],
		resample_fps=float(config["windowing"]["sampling_fps"]),
		max_gap_seconds=float(config["windowing"]["max_gap_seconds"]),
		max_missing_fraction=float(config["windowing"]["max_missing_fraction"]),
	)
	subjects = sorted({item[2] for item in dataset})
	test_subject = args.test_subject or subjects[-1]
	train_data, val_data = split_by_subject(dataset, test_subject)
	model_builders = {
		"cnn": build_cnn_model,
		"ann": build_ann_model,
		"dnn": build_dnn_model,
		"cnn_lstm": build_cnn_lstm_model,
		"dnn_lstm": build_dnn_lstm_model,
	}
	model_builder = model_builders[args.model_type]
	if args.model_type in ("cnn", "cnn_lstm"):
		model = model_builder(input_features=3, num_classes=len(config["classes"]))
	else:
		model = model_builder(
			input_features=3,
			frames=window_size,
			num_classes=len(config["classes"]),
		)
	history = train_model(
		model, train_data, val_data, args.epochs,
		learning_rate=args.learning_rate, batch_size=args.batch_size,
	)
	save_model(model, args.model_path)
	print(f"Model: {args.model_type.upper()}")
	print(f"Held-out subject: {test_subject}")
	print(f"Train windows: {len(train_data)}; validation windows: {len(val_data)}")
	print(f"Initial loss: {history['train_loss'][0]:.4f}")
	print(f"Final loss: {history['train_loss'][-1]:.4f}")
	print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2%}")
	print(f"Saved model: {args.model_path}")


if __name__ == "__main__":
	main()
