"""Build the combined feature CSV from the recorded videos.

The script keeps command-line handling small: the extraction pipeline owns
video discovery and feature computation, while this file only supplies paths
and starts it.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.features.extract_features import build_features_csv


def main():
	# Defaults point to the project's standard raw-video and feature locations.
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-raw-dir", type=Path, default=ROOT / "data" / "raw")
	parser.add_argument("--output-csv", type=Path, default=ROOT / "data" / "features.csv")
	args = parser.parse_args()
	build_features_csv(args.data_raw_dir, args.output_csv)


if __name__ == "__main__":
	main()
