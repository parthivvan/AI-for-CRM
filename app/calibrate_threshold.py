"""
Threshold calibration for the GenX 360 local ONNX skin/scalp classifier.

WHAT THIS DOES
---------------
1. Runs the model on your REAL labeled test set (ImageFolder layout:
   <test_dir>/<label_name>/*.jpg) and records confidence on every
   correct prediction.
2. Generates a battery of synthetic "garbage" inputs (random noise,
   flat colors, heavy blur, extreme brightness) that no real skin
   photo should ever resemble, and records the model's confidence
   on those.
3. Sweeps candidate thresholds and reports, at each one:
     - % of correct real predictions RETAINED (i.e. not needlessly
       kicked to the deterministic fallback)
     - % of garbage inputs FALSELY ACCEPTED as confident real labels
4. Suggests a threshold: the lowest value where garbage false-accept
   rate drops under a target ceiling (default 5%), and shows what
   real-accuracy cost that choice has, if any.

HONEST LIMITATION
------------------
Synthetic noise/color/blur images are a proxy for "not a real skin
photo," not a substitute for real bad photos (dark room, wrong body
part, heavy motion blur, screenshots, etc). If you have any real
examples of bad uploads, drop them in a folder and pass
--real-garbage-dir to include them — that result is more trustworthy
than the synthetic battery alone.

USAGE
-----
python calibrate_threshold.py \
    --model mobilenetv3_large_skin_ai.onnx \
    --labels labels.json \
    --test-dir "/path/to/Skin_AI_Dataset/test" \
    --real-garbage-dir "/path/to/known_bad_photos"   # optional
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

_INPUT_SIZE = (224, 224)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """Must exactly match app/local_model_provider.py's _preprocess."""
    img = img.convert("RGB").resize(_INPUT_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - _MEAN) / _STD
    return arr[np.newaxis, ...]


class Model:
    def __init__(self, model_path: str, labels_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))

    def predict(self, img: Image.Image) -> tuple[str, float]:
        tensor = preprocess_pil(img)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])


def load_real_test_set(test_dir: Path, model: Model) -> list[dict]:
    """ImageFolder layout: test_dir/<label>/*.jpg|png"""
    records = []
    if not test_dir.exists():
        print(f"WARNING: test dir {test_dir} does not exist — skipping real-set evaluation.")
        return records

    for label_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        true_label = label_dir.name
        for img_path in sorted(label_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            try:
                img = Image.open(img_path)
                pred_label, conf = model.predict(img)
            except Exception as exc:
                print(f"  skip {img_path.name}: {exc}")
                continue
            records.append(
                {
                    "path": str(img_path),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": conf,
                    "correct": pred_label == true_label,
                }
            )
    return records


def generate_synthetic_garbage(n_noise: int = 30, seed: int = 0) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    images: list[Image.Image] = []

    # Pure random noise
    for _ in range(n_noise):
        arr = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
        images.append(Image.fromarray(arr, mode="RGB"))

    # Flat solid colors across a spread of the color space
    for r in (0, 64, 128, 192, 255):
        for g in (0, 128, 255):
            for b in (0, 128, 255):
                arr = np.ones((224, 224, 3), dtype=np.uint8) * np.array([r, g, b], dtype=np.uint8)
                images.append(Image.fromarray(arr, mode="RGB"))

    # Heavily blurred noise (simulates an out-of-focus / low-quality upload)
    for _ in range(10):
        arr = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB").filter(ImageFilter.GaussianBlur(radius=15))
        images.append(img)

    # Extreme brightness / near-black / near-white (bad lighting)
    for val in (5, 250):
        arr = np.ones((224, 224, 3), dtype=np.uint8) * val
        images.append(Image.fromarray(arr, mode="RGB"))

    return images


def load_real_garbage(dir_path: Path | None) -> list[Image.Image]:
    if dir_path is None:
        return []
    if not dir_path.exists():
        print(f"WARNING: real-garbage dir {dir_path} does not exist — skipping.")
        return []
    images = []
    for img_path in sorted(dir_path.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            images.append(Image.open(img_path))
        except Exception as exc:
            print(f"  skip {img_path.name}: {exc}")
    return images


def sweep_thresholds(
    real_records: list[dict],
    garbage_confidences: list[float],
    thresholds: list[float],
) -> list[dict]:
    correct_confidences = [r["confidence"] for r in real_records if r["correct"]]
    total_correct = len(correct_confidences)
    total_garbage = len(garbage_confidences)

    rows = []
    for t in thresholds:
        retained = sum(1 for c in correct_confidences if c >= t)
        false_accept = sum(1 for c in garbage_confidences if c >= t)
        rows.append(
            {
                "threshold": t,
                "real_retained_pct": round(100 * retained / total_correct, 1) if total_correct else None,
                "garbage_false_accept_pct": round(100 * false_accept / total_garbage, 1) if total_garbage else None,
            }
        )
    return rows


def suggest_threshold(rows: list[dict], target_false_accept_pct: float = 5.0) -> dict | None:
    candidates = [r for r in rows if r["garbage_false_accept_pct"] is not None and r["garbage_false_accept_pct"] <= target_false_accept_pct]
    if not candidates:
        return None
    # lowest threshold that meets the false-accept ceiling = best real-accuracy retention
    return min(candidates, key=lambda r: r["threshold"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--test-dir", required=True, help="ImageFolder layout: test_dir/<label>/*.jpg")
    parser.add_argument("--real-garbage-dir", default=None, help="Optional folder of known-bad real photos")
    parser.add_argument("--n-noise", type=int, default=30)
    parser.add_argument("--target-false-accept-pct", type=float, default=5.0)
    parser.add_argument("--out", default="calibration_report.json")
    args = parser.parse_args()

    model = Model(args.model, args.labels)

    print(f"Loading real test set from {args.test_dir} ...")
    real_records = load_real_test_set(Path(args.test_dir), model)
    n_correct = sum(1 for r in real_records if r["correct"])
    print(f"  {len(real_records)} real images, {n_correct} correct ({100*n_correct/len(real_records):.1f}% acc)" if real_records else "  no real images found")

    print("Generating synthetic garbage battery ...")
    garbage_images = generate_synthetic_garbage(n_noise=args.n_noise)
    real_garbage_images = load_real_garbage(Path(args.real_garbage_dir) if args.real_garbage_dir else None)
    all_garbage_images = garbage_images + real_garbage_images
    garbage_confidences = [model.predict(img)[1] for img in all_garbage_images]
    print(f"  {len(garbage_images)} synthetic + {len(real_garbage_images)} real-garbage = {len(all_garbage_images)} total")

    thresholds = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
    rows = sweep_thresholds(real_records, garbage_confidences, thresholds)

    print(f"\n{'Threshold':>10} | {'Real Retained %':>16} | {'Garbage False-Accept %':>23}")
    print("-" * 55)
    for r in rows:
        print(f"{r['threshold']:>10} | {str(r['real_retained_pct']):>16} | {str(r['garbage_false_accept_pct']):>23}")

    suggestion = suggest_threshold(rows, args.target_false_accept_pct)
    print()
    if suggestion:
        print(
            f"SUGGESTED threshold: {suggestion['threshold']} "
            f"(garbage false-accept {suggestion['garbage_false_accept_pct']}% <= target {args.target_false_accept_pct}%, "
            f"retains {suggestion['real_retained_pct']}% of correct real predictions)"
        )
    else:
        print(
            f"No threshold in the sweep range gets garbage false-accept under {args.target_false_accept_pct}%. "
            "Consider a higher threshold ceiling in the sweep, or revisit the model (an explicit "
            "'other_or_unclear' training class would fix this at the model level rather than the threshold level)."
        )

    report = {
        "current_config_threshold": 0.65,
        "real_test_accuracy": (n_correct / len(real_records)) if real_records else None,
        "n_real_images": len(real_records),
        "n_garbage_images": len(all_garbage_images),
        "threshold_sweep": rows,
        "suggested_threshold": suggestion,
        "per_class_note": "This script reports overall retained/false-accept rates. Check redness specifically given its known precision issue.",
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
