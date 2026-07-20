"""dataset.py — Validate CSV labels and produce train/val/test splits.

Usage:
    python training/dataset.py \
        --csv data/labels.csv \
        --images data/images/ \
        --out data/splits/

CSV columns required:
    image_path, image_type, pigmentation, redness, uneven_texture,
    dryness, hair_density, other_or_unclear
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LABEL_COLS = [
    "pigmentation",
    "redness",
    "uneven_texture",
    "dryness",
    "hair_density",
    "other_or_unclear",
]
REQUIRED_COLS = {"image_path", "image_type"} | set(LABEL_COLS)


def validate_and_split(
    csv_path: Path,
    images_root: Path,
    out_dir: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> None:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"ERROR: CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Validate image files exist
    bad = [row.image_path for row in df.itertuples() if not (images_root / row.image_path).exists()]
    if bad:
        print(f"ERROR: {len(bad)} image(s) not found under {images_root}. First 5: {bad[:5]}", file=sys.stderr)
        sys.exit(1)

    # Validate binary labels
    for col in LABEL_COLS:
        bad_vals = df[~df[col].isin([0, 1])][col]
        if not bad_vals.empty:
            print(f"ERROR: column {col!r} contains non-binary values.", file=sys.stderr)
            sys.exit(1)

    # Split
    test_frac = 1.0 - train_frac - val_frac
    train_df, temp_df = train_test_split(df, test_size=(1 - train_frac), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_frac / (val_frac + test_frac), random_state=42)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    summary = {
        "total": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "labels": LABEL_COLS,
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Split complete: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split GenX 360 label CSV.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()
    validate_and_split(args.csv, args.images, args.out, args.train_frac, args.val_frac)


if __name__ == "__main__":
    main()
