"""export_onnx.py — Export PyTorch MobileNetV3-Large model to ONNX format.

Target Output Files:
    models/current/mobilenetv3_large_skin_ai.onnx
    models/current/labels.json

Usage:
    python training/export_onnx.py \
        --checkpoint checkpoints/best_mobilenetv3.pth \
        --out models/current/ \
        --version genx-mobilenetv3-v1.0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

LABEL_COLS: list[str] = [
    "acne",
    "dryness",
    "hair_thinning",
    "pigmentation",
    "redness",
]


def export(checkpoint: Path, out_dir: Path, version: str = "genx-mobilenetv3-v1.0") -> None:
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_large

    device = torch.device("cpu")  # Export on CPU for portability

    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(LABEL_COLS))

    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint weights from {checkpoint}")
    else:
        print(f"WARNING: Checkpoint {checkpoint} not found. Exporting default initialized architecture.")

    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "mobilenetv3_large_skin_ai.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"ONNX model successfully written to {onnx_path}")

    # Write labels.json
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(LABEL_COLS, indent=2), encoding="utf-8")
    print(f"Labels config written to {labels_path}")

    # Also sync root models/skin_analysis.onnx for backwards compatibility
    root_model_path = Path("models/skin_analysis.onnx")
    root_model_path.write_bytes(onnx_path.read_bytes())
    print(f"Synced copy to {root_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MobileNetV3-Large Checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_mobilenetv3.pth"))
    parser.add_argument("--out", type=Path, default=Path("models/current"))
    parser.add_argument("--version", type=str, default="genx-mobilenetv3-v1.0")
    args = parser.parse_args()

    export(args.checkpoint, args.out, args.version)


if __name__ == "__main__":
    main()
