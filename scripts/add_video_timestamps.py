"""Add source-video FPS and timestamps to an already extracted feature CSV.

This is equivalent to recording the metadata during extraction when the CSV
already retains the original frame number for every detected frame.
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def source_fps_by_recording(raw_dir: Path) -> dict[tuple[str, int], float]:
    labels = {"TALKING": 0, "LOOKING": 1, "READING": 2}
    fps_by_recording = {}
    for path in raw_dir.glob("*.mp4"):
        subject_id, activity = path.stem.rsplit("_", 1)
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if fps <= 0:
            raise ValueError(f"Could not determine FPS for {path}")
        fps_by_recording[(subject_id, labels[activity])] = fps
    return fps_by_recording


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data" / "features_six_subjects.csv")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "data" / "features_7_5fps.csv")
    args = parser.parse_args()

    frame_data = pd.read_csv(args.input_csv)
    required = {"subject_id", "class_label", "frame_number", "pitch", "yaw", "gaze_ratio"}
    missing = required.difference(frame_data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    fps_map = source_fps_by_recording(args.raw_dir)
    keys = list(zip(frame_data["subject_id"], frame_data["class_label"].astype(int)))
    unknown = sorted(set(keys).difference(fps_map))
    if unknown:
        raise ValueError(f"No source video FPS found for: {unknown}")

    frame_data["source_fps"] = [fps_map[key] for key in keys]
    frame_data["timestamp_seconds"] = frame_data["frame_number"] / frame_data["source_fps"]
    ordered_columns = [
        "subject_id", "class_label", "frame_number", "timestamp_seconds",
        "source_fps", "pitch", "yaw", "gaze_ratio",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame_data[ordered_columns].to_csv(args.output_csv, index=False)
    print(f"Wrote {len(frame_data):,} timestamped rows to {args.output_csv}")


if __name__ == "__main__":
    main()
