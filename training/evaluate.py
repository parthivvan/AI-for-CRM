"""evaluate.py — Evaluate 5-class MobileNetV3-Large model on test/val/train splits.

Generates evaluation report with precision, recall, f1-score, and support per class.

Usage:
    python training/evaluate.py \
        --dataset-dir data/Skin_AI_Dataset/ \
        --splits-dir data/splits/ \
        --checkpoint checkpoints/best_mobilenetv3.pth \
        --out scratch/onnx_evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

LABEL_COLS: list[str] = [
    "acne",
    "dryness",
    "hair_thinning",
    "pigmentation",
    "redness",
]


def evaluate_split(model, dataset, loader, device) -> dict:
    import numpy as np
    import torch
    from sklearn.metrics import classification_report

    model.eval()
    all_preds, all_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    elapsed = time.time() - start_time
    total = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = round(correct / max(total, 1), 4)

    metrics_per_class = {}
    for idx, label in enumerate(LABEL_COLS):
        label_indices = [i for i, l in enumerate(all_labels) if l == idx]
        support = len(label_indices)
        correct_class = sum(all_preds[i] == idx for i in label_indices)
        pred_class_count = sum(p == idx for p in all_preds)

        precision = round(correct_class / max(pred_class_count, 1), 4)
        recall = round(correct_class / max(support, 1), 4)
        f1 = round(2 * (precision * recall) / max(precision + recall, 1e-6), 4)

        metrics_per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
            "correct": correct_class,
        }

    return {
        "accuracy": accuracy,
        "total_images": total,
        "elapsed_seconds": round(elapsed, 2),
        "ms_per_image": round((elapsed / max(total, 1)) * 1000, 2),
        "metrics_per_class": metrics_per_class,
    }


def evaluate_all(dataset_dir: Path, splits_dir: Path, checkpoint: Path, out_file: Path) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import mobilenet_v3_large

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train import GenXSingleLabelDataset, build_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(LABEL_COLS))

    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    model = model.to(device)

    report = {}
    for split in ("test", "validation", "train"):
        split_json = splits_dir / f"{'val' if split == 'validation' else split}.json"
        if not split_json.exists():
            continue
        ds = GenXSingleLabelDataset(split_json, dataset_dir, build_transforms(train=False))
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        report[split] = evaluate_split(model, ds, loader, device)

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Evaluation report saved to {out_file}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GenX MobileNetV3-Large Classifier")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/Skin_AI_Dataset"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_mobilenetv3.pth"))
    parser.add_argument("--out", type=Path, default=Path("scratch/onnx_evaluation_report.json"))
    args = parser.parse_args()

    evaluate_all(args.dataset_dir, args.splits_dir, args.checkpoint, args.out)


if __name__ == "__main__":
    main()
