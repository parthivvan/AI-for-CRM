"""dataset.py — Validate dataset structure and produce train/val/test splits.

Supports both:
1. ImageFolder directory layout (<dataset_root>/<label_name>/*.jpg)
2. CSV-based label files

Target 5-class label set:
    ['acne', 'dryness', 'hair_thinning', 'pigmentation', 'redness']

Usage:
    python training/dataset.py \
        --dataset-dir data/Skin_AI_Dataset/ \
        --out data/splits/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

LABEL_COLS: list[str] = [
    "acne",
    "dryness",
    "hair_thinning",
    "pigmentation",
    "redness",
]
REQUIRED_COLS = {"image_path", "image_type"} | set(LABEL_COLS)


def validate_and_split_folder(
    dataset_dir: Path,
    out_dir: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> None:
    """Split an ImageFolder structured directory into train/val/test splits."""
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, str]] = []

    for label in LABEL_COLS:
        class_dir = dataset_dir / label
        if not class_dir.exists():
            print(f"WARNING: Class directory {class_dir} not found.", file=sys.stderr)
            continue

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            for img_path in class_dir.glob(ext):
                rel_path = str(img_path.relative_to(dataset_dir))
                samples.append({
                    "image_path": rel_path,
                    "label": label,
                    "image_type": "scalp" if label == "hair_thinning" else "skin",
                })

    if not samples:
        print(f"ERROR: No valid images found under {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    random.seed(42)
    random.shuffle(samples)

    total = len(samples)
    n_train = int(total * train_frac)
    n_val = int(total * val_frac)

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for split_name, split_samples in splits.items():
        out_file = out_dir / f"{split_name}.json"
        out_file.write_text(json.dumps(split_samples, indent=2), encoding="utf-8")
        print(f"Wrote {len(split_samples)} samples to {out_file}")

    labels_file = out_dir / "labels.json"
    labels_file.write_text(json.dumps(LABEL_COLS, indent=2), encoding="utf-8")
    print(f"Wrote label configuration to {labels_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GenX 360 Dataset Validator & Splitter")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/Skin_AI_Dataset"), help="Path to ImageFolder dataset")
    parser.add_argument("--out", type=Path, default=Path("data/splits"), help="Output directory for split JSON files")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Train split fraction")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation split fraction")
    args = parser.parse_args()

    validate_and_split_folder(args.dataset_dir, args.out, args.train_frac, args.val_frac)


if __name__ == "__main__":
    main()
