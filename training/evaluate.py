"""evaluate.py — Per-label precision, recall, F1 on the test split.

Usage:
    python training/evaluate.py \
        --splits data/splits/ \
        --images data/images/ \
        --checkpoint checkpoints/best.pth \
        --threshold 0.65
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL_COLS = [
    "pigmentation",
    "redness",
    "uneven_texture",
    "dryness",
    "hair_density",
    "other_or_unclear",
]


def evaluate(
    splits_dir: Path,
    images_root: Path,
    checkpoint: Path,
    threshold: float = 0.65,
) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report
    from torch.utils.data import DataLoader
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    # Reuse dataset class from train.py
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train import GenXDataset, build_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABEL_COLS))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    test_ds = GenXDataset(splits_dir / "test.csv", images_root, build_transforms(train=False))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = model(imgs.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.numpy())

    y_pred = (np.concatenate(all_preds) >= threshold).astype(int)
    y_true = np.concatenate(all_labels).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_COLS,
        output_dict=True,
        zero_division=0,
    )

    out_path = splits_dir.parent / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Evaluation report written to {out_path}")
    print(classification_report(y_true, y_pred, target_names=LABEL_COLS, zero_division=0))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GenX 360 classifier on the test split.")
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    evaluate(args.splits, args.images, args.checkpoint, args.threshold)


if __name__ == "__main__":
    main()
