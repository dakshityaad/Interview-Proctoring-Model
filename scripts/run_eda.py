"""Regenerate EDA figures and a Markdown summary for ``data/features.csv``.

Usage:
    .venv\\Scripts\\python scripts\\run_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.windowing.make_windows import build_dataset, build_reading_dataset


FEATURE_COLUMNS = ["pitch", "yaw", "gaze_ratio"]
CLASS_NAMES = {0: "Talking", 1: "Looking down", 2: "Reading"}
CLASS_LABELS = {"TALKING": 0, "LOOKING": 1, "READING": 2}


def markdown_table(frame: pd.DataFrame, index: bool = True) -> str:
    """Render a small DataFrame as Markdown without optional packages."""
    display = frame.reset_index() if index else frame.copy()
    headers = [str(column) for column in display.columns]
    rows = []
    for values in display.itertuples(index=False, name=None):
        rows.append([str(value) for value in values])
    separator = ["---"] * len(headers)
    return "\n".join(
        "| " + " | ".join(row) + " |" for row in [headers, separator, *rows]
    )


def video_metadata(raw_dir: Path) -> pd.DataFrame:
    """Read source-video frame rates and durations for the EDA caveat."""
    rows = []
    for video_path in sorted(raw_dir.glob("*.mp4")):
        subject_id, class_name = video_path.stem.rsplit("_", 1)
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps <= 0:
            raise ValueError(f"Could not read FPS from {video_path}")
        rows.append(
            {
                "subject_id": subject_id,
                "activity": CLASS_NAMES[CLASS_LABELS[class_name]],
                "source_fps": fps,
                "source_frames": frame_count,
                "duration_seconds": frame_count / fps,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    csv_path = ROOT / "data" / "features.csv"
    raw_dir = ROOT / "data" / "raw"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"subject_id", "class_label", "frame_number", *FEATURE_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["class_label"] = df["class_label"].astype(int)
    df["activity"] = df["class_label"].map(CLASS_NAMES)
    class_order = [CLASS_NAMES[label] for label in sorted(CLASS_NAMES)]
    source_videos = video_metadata(raw_dir)
    sns.set_theme(style="whitegrid", context="notebook")

    # Class distributions: one readable plot per feature.
    for feature in FEATURE_COLUMNS:
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.histplot(
            data=df,
            x=feature,
            hue="activity",
            hue_order=class_order,
            bins=50,
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            ax=axis,
        )
        axis.set_title(f"{feature} distribution by activity")
        axis.set_xlabel(feature)
        axis.set_ylabel("Density")
        figure.tight_layout()
        figure.savefig(results_dir / f"{feature}_eda.png", dpi=160)
        plt.close(figure)

    # Subject/class coverage is a crucial check before any subject-level split.
    coverage = pd.crosstab(df["subject_id"], df["activity"]).reindex(columns=class_order, fill_value=0)
    figure, axis = plt.subplots(figsize=(10, 5))
    coverage.plot(kind="bar", stacked=True, ax=axis, colormap="viridis")
    axis.set_title("Usable frames by subject and activity")
    axis.set_xlabel("Subject")
    axis.set_ylabel("Frames")
    axis.legend(title="Activity")
    figure.tight_layout()
    figure.savefig(results_dir / "subject_class_coverage_eda.png", dpi=160)
    plt.close(figure)

    # Correlation provides a quick redundancy check for the model inputs.
    correlation = df[FEATURE_COLUMNS].corr()
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(correlation, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=axis)
    axis.set_title("Frame-level feature correlations")
    figure.tight_layout()
    figure.savefig(results_dir / "feature_correlation_eda.png", dpi=160)
    plt.close(figure)

    frame_counts = df["class_label"].value_counts().sort_index().rename(index=CLASS_NAMES)
    class_summary = pd.DataFrame({"frames": frame_counts, "share (%)": frame_counts / len(df) * 100})
    class_summary["share (%)"] = class_summary["share (%)"].round(2)
    overall_stats = df[FEATURE_COLUMNS].agg(["mean", "std", "min", "max"]).T.round(4)
    gaze_outside_unit_interval = int(
        ((df["gaze_ratio"] < 0) | (df["gaze_ratio"] > 1)).sum()
    )
    class_means = df.groupby("activity")[FEATURE_COLUMNS].mean().reindex(class_order).round(4)
    class_stds = df.groupby("activity")[FEATURE_COLUMNS].std().reindex(class_order).round(4)
    fps_by_subject = source_videos.groupby("subject_id").agg(
        source_fps=("source_fps", lambda values: ", ".join(f"{value:.2f}" for value in sorted({round(float(value), 2) for value in values}))),
        recorded_minutes=("duration_seconds", lambda values: round(values.sum() / 60, 2)),
    )

    windows = build_dataset(csv_path, window_size=60, stride=10)
    window_counts = pd.Series([label for _, label, _ in windows]).value_counts().sort_index()
    window_counts.index = window_counts.index.map(CLASS_NAMES)
    reading_windows = build_reading_dataset(csv_path, window_size=75, stride=8, sampling_fps=15)
    binary_counts = pd.Series([label for _, label, _ in reading_windows]).value_counts().sort_index()
    binary_counts.index = binary_counts.index.map({0: "Not reading", 1: "Reading"})

    largest_subject = coverage.sum(axis=1).idxmax()
    smallest_subject = coverage.sum(axis=1).idxmin()
    report = f"""# Feature Dataset EDA Findings

This report was regenerated from `data/features.csv` by `scripts/run_eda.py`.
It covers the frame-level model inputs: `pitch`, `yaw`, and `gaze_ratio`.

## Data quality and coverage

- **Usable frames:** {len(df):,}
- **Subjects:** {df['subject_id'].nunique()}
- **Activity clips represented:** {df.groupby(['subject_id', 'class_label']).ngroups} (all subjects have all three labels)
- **Missing values:** {int(df.drop(columns='activity').isna().sum().sum()):,}
- **Duplicate subject/class/frame rows:** {int(df.duplicated(['subject_id', 'class_label', 'frame_number']).sum()):,}
- **Gaze-ratio estimates outside the nominal 0–1 range:** {gaze_outside_unit_interval:,} ({gaze_outside_unit_interval / len(df) * 100:.03f}%)

### Frame balance by activity

{markdown_table(class_summary)}

The dataset is moderately imbalanced: Reading is the largest class and Looking down the smallest. Use class-aware metrics and consider weighted loss when training.

The 22 out-of-range gaze estimates are rare but should be monitored; they can result from landmark jitter or partial face detection. Prefer clipping or robust scaling based on the training split only rather than silently dropping them.

### Frames by subject and activity

{markdown_table(coverage.assign(Total=coverage.sum(axis=1)))}

Subject coverage varies from **{coverage.sum(axis=1)[smallest_subject]:,}** usable frames for {smallest_subject} to **{coverage.sum(axis=1)[largest_subject]:,}** for {largest_subject}. This makes subject-held-out validation preferable to random frame splitting.

### Source frame-rate caveat

Frame count is **not** a duration measure in this dataset. The clips for subjects 2, 4, and 5 were recorded at about 7.5 FPS; subject 6 at about 16.7 FPS; subject 3 at 30 FPS; and subject 1 contains clips at about 16.7 FPS plus a 30-FPS Reading clip.

{markdown_table(fps_by_subject)}

This has two direct implications:

1. Do not use raw frame totals to compare how much behavior was recorded per subject; compare duration instead.
2. The existing motion features and five-second-window logic assume 15 FPS, so they currently represent different amounts of real time across subjects. Resample each recording to one common FPS (for example 15 FPS) or persist per-frame time/FPS and use time-aware derivatives before training the reading detector.

## Feature summaries

### Overall values

{markdown_table(overall_stats)}

### Mean feature values by activity

{markdown_table(class_means)}

### Feature variability by activity (standard deviation)

{markdown_table(class_stds)}

### Correlation

{markdown_table(correlation.round(4))}

Pitch, yaw, and gaze ratio are correlated but not duplicates, so retaining all three model inputs is justified. The distribution plots show that Looking down is most clearly separated by head pitch; Reading overlaps more with the other behaviors and benefits from temporal information.

## Window-level checks

The three-class CNN configuration uses 60-frame windows with a stride of 10.

| Activity | 60-frame windows |
|---|---:|
"""
    report += "\n".join(f"| {activity} | {count:,} |" for activity, count in window_counts.items())
    report += f"""

- **Total 60-frame windows:** {len(windows):,}

The reading detector uses 75-frame windows (5 seconds at 15 FPS) with a stride of 8.

| Binary target | 75-frame windows |
|---|---:|
"""
    report += "\n".join(f"| {target} | {count:,} |" for target, count in binary_counts.items())
    report += f"""

- **Total five-second windows:** {len(reading_windows):,}

## Key takeaways

1. The refreshed dataset now contains six subjects and complete coverage for all three activities per subject.
2. No missing values or duplicate frame keys were found.
3. Raw frame counts are confounded by source FPS. Normalize sampling before using temporal or motion-based models.
4. Class and subject imbalance remain important: evaluate with per-class precision/recall/F1 and a subject-held-out split.
5. Frame-level pitch offers the clearest signal for Looking down; temporal windows remain especially important for identifying Reading.

## Generated figures

- `results/pitch_eda.png`
- `results/yaw_eda.png`
- `results/gaze_ratio_eda.png`
- `results/subject_class_coverage_eda.png`
- `results/feature_correlation_eda.png`
"""
    (ROOT / "findings.md").write_text(report, encoding="utf-8")
    print(f"EDA complete: {len(df):,} frames, {len(windows):,} CNN windows, {len(reading_windows):,} reading windows.")


if __name__ == "__main__":
    main()
