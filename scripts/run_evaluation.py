"""Evaluate a saved CNN, ANN, or DNN checkpoint on one held-out subject."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.ann_model import build_model as build_ann_model
from src.model.cnn_model import build_model as build_cnn_model
from src.model.dataset import split_by_subject, to_tensors
from src.model.dnn_model import build_model as build_dnn_model
from src.model.cnn_lstm_model import build_model as build_cnn_lstm_model
from src.model.dnn_lstm_model import build_model as build_dnn_lstm_model
from src.windowing.make_windows import build_dataset, build_reading_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-type", choices=("cnn", "ann", "dnn", "cnn_lstm", "dnn_lstm"), required=True)
    parser.add_argument("--test-subject", required=True)
    parser.add_argument("--reading", action="store_true", help="Evaluate the binary five-second reading detector")
    args = parser.parse_args()

    with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    windowing = config["windowing"]
    if args.reading:
        reading = config["reading_detection"]
        fps = float(reading["sampling_fps"])
        window_size = round(float(reading["window_seconds"])
                            * fps)
        dataset = build_reading_dataset(
            args.features_csv,
            window_size=window_size,
            stride=max(1, round(float(reading["stride_seconds"])
                                * fps)),
            sampling_fps=fps,
            max_gap_seconds=float(windowing["max_gap_seconds"]),
            max_missing_fraction=float(windowing["max_missing_fraction"]),
        )
        num_classes = 2
        input_features = 10
    else:
        window_size = windowing["window_size"]
        dataset = build_dataset(
            args.features_csv,
            window_size=window_size,
            stride=windowing["stride"],
            resample_fps=float(windowing["sampling_fps"]),
            max_gap_seconds=float(windowing["max_gap_seconds"]),
            max_missing_fraction=float(windowing["max_missing_fraction"]),
        )
        num_classes = 3
        input_features = 3
    _, validation_data = split_by_subject(dataset, args.test_subject)
    builders = {
        "cnn": build_cnn_model,
        "ann": build_ann_model,
        "dnn": build_dnn_model,
        "cnn_lstm": build_cnn_lstm_model,
        "dnn_lstm": build_dnn_lstm_model,
    }
    if args.model_type in ("cnn", "cnn_lstm"):
        model = builders[args.model_type](input_features=input_features, num_classes=num_classes)
    else:
        model = builders[args.model_type](input_features=input_features, frames=window_size, num_classes=num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
    model.eval()

    features, labels = to_tensors(validation_data)
    with torch.no_grad():
        probabilities = model(features).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    expected = labels.numpy()

    metric_labels = [0, 1] if args.reading else [0, 1, 2]
    confusion = confusion_matrix(expected, predictions, labels=metric_labels)
    class_names = ["Not reading", "Reading"] if args.reading else ["Talking", "Looking down", "Reading"]
    report = classification_report(
        expected,
        predictions,
        labels=metric_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(f"Model: {args.model_type.upper()}")
    print(f"Held-out subject: {args.test_subject}")
    print(f"Windows evaluated: {len(expected)}")
    print(f"Accuracy: {accuracy_score(expected, predictions):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(expected, predictions):.4f}")
    print(f"Macro precision: {precision_score(expected, predictions, average='macro', zero_division=0):.4f}")
    print(f"Macro recall: {recall_score(expected, predictions, average='macro', zero_division=0):.4f}")
    print(f"Macro F1: {f1_score(expected, predictions, average='macro', zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(expected, predictions, average='weighted', zero_division=0):.4f}")
    print(f"Log loss: {log_loss(expected, probabilities, labels=metric_labels):.4f}")
    try:
        if args.reading:
            print(f"ROC AUC: {roc_auc_score(expected, probabilities[:, 1]):.4f}")
        else:
            print(f"ROC AUC (OvR macro): {roc_auc_score(expected, probabilities, multi_class='ovr', average='macro'):.4f}")
            print(f"ROC AUC (OvR weighted): {roc_auc_score(expected, probabilities, multi_class='ovr', average='weighted'):.4f}")
    except ValueError as exc:
        print(f"ROC AUC: unavailable ({exc})")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion)
    print("Per-class metrics:")
    print(report)


if __name__ == "__main__":
    main()
