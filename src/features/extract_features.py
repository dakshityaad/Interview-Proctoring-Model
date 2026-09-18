"""Extract frame-level face features from the recorded training videos.

The idea in plain English:
    1. Open each recorded video and read it one frame at a time.
    2. Ask MediaPipe for the landmarks of the first face in each frame.
    3. Turn those landmarks into head pose and gaze measurements.
    4. Store one labeled row per usable frame for model training.

Frames without a detected face are skipped. A missing-face row would contain
no meaningful pose or gaze values and would introduce artificial gaps into the
feature space instead of describing the candidate's behavior.
"""

import csv
from pathlib import Path

from src.capture import webcam_stream
from src.landmarks import face_mesh, gaze, head_pose


CSV_COLUMNS = [
    "subject_id",
    "class_label",
    "frame_number",
    "pitch",
    "yaw",
    "gaze_ratio",
]
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mkv", ".webm"}
CLASS_LABELS = {"TALKING": 0, "LOOKING": 1, "READING": 2}


def _class_label_from_path(video_path):
    """Convert a class video name into the label used by the classifiers."""
    # Numeric names already match the documented class IDs: 0, 1, and 2.
    label = video_path.stem
    try:
        return int(label)
    except ValueError:
        return CLASS_LABELS.get(label.upper(), label)


def _video_jobs(data_raw_dir):
    """Yield ``(video_path, subject_id, class_label)`` for supported layouts."""
    # The documented layout is data/raw/<subject>/<class video>.
    subject_dirs = sorted(path for path in data_raw_dir.iterdir() if path.is_dir())
    for subject_dir in subject_dirs:
        for video_path in sorted(
            path
            for path in subject_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ):
            yield video_path, subject_dir.name, _class_label_from_path(video_path)

    # Also accept the flat recording layout, such as SUBJECT_01_READING.mp4.
    for video_path in sorted(
        path
        for path in data_raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ):
        subject_id, class_name = video_path.stem.rsplit("_", 1)
        yield video_path, subject_id, CLASS_LABELS.get(class_name.upper(), class_name)


def extract_features_from_video(path, subject_id, class_label):
    """Return ``[subject, class, frame, pitch, yaw, gaze]`` rows for one video."""
    rows = []
    # The detector is initialized once per video, not once per frame.
    detector = face_mesh.init_face_mesh()

    for frame_number, frame in enumerate(webcam_stream.read_video_frames(path)):
        # Face landmarks are the shared input for both downstream estimators.
        landmarks = face_mesh.get_landmarks(detector, frame)
        if landmarks is None:
            # Keep the original frame number so missing detections remain visible
            # as gaps when the extracted CSV is sanity-checked.
            continue

        frame_height, frame_width = frame.shape[:2]
        # Pose needs the image dimensions to convert normalized landmarks to pixels.
        pitch, yaw = head_pose.get_head_pose(
            landmarks, frame_width, frame_height
        )
        if pitch is None or yaw is None:
            continue

        # The model uses the horizontal component as its single gaze feature;
        # vertical movement is represented separately by head pitch.
        gaze_ratio = gaze.get_gaze_ratio(landmarks, frame_width, frame_height)[0]
        rows.append(
            [
                subject_id,
                class_label,
                frame_number,
                pitch,
                yaw,
                gaze_ratio,
            ]
        )

    return rows


def build_features_csv(data_raw_dir, output_csv_path):
    """Extract every subject/class video into one CSV and return its rows."""
    data_raw_dir = Path(data_raw_dir)
    output_csv_path = Path(output_csv_path)
    rows = []

    # Process videos in stable order so repeated extraction runs are comparable.
    for video_path, subject_id, class_label in _video_jobs(data_raw_dir):
        rows.extend(extract_features_from_video(video_path, subject_id, class_label))

    # Create the destination folder when callers choose a nested output path.
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)

    return rows