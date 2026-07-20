"""export_onnx.py — Export a trained PyTorch checkpoint to ONNX.

Usage:
    python training/export_onnx.py \
        --checkpoint checkpoints/best.pth \
        --out models/current/ \
        --version genx-vision-v0.1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

LABEL_COLS = [
    "pigmentation",
    "redness",
    "uneven_texture",
    "dryness",
    "hair_density",
    "other_or_unclear",
]


def export(checkpoint: Path, out_dir: Path, version: str = "genx-vision-v0.1") -> None:
    import torch
    import torch.nn as nn
    from torchvision.models import efficientnet_b0

    device = torch.device("cpu")  # Always export from CPU for portability

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABEL_COLS))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"ONNX model written to {onnx_path}")

    # Write labels.json (overwrites existing)
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(LABEL_COLS))

    # Write model metadata
    metadata = {
        "model_version": version,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint),
        "labels": LABEL_COLS,
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, len(LABEL_COLS)],
        "output_activation": "sigmoid",
        "architecture": "efficientnet_b0",
        "opset_version": 17,
    }
    metadata_path = out_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata written to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GenX 360 classifier to ONNX.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--version", default="genx-vision-v0.1")
    args = parser.parse_args()
    export(args.checkpoint, args.out, args.version)


if __name__ == "__main__":
    main()
