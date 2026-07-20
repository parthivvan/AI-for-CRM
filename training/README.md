# GenX 360 Vision Model Training Pipeline

Fine-tune EfficientNet-B0 as a multi-label skin/scalp/hair image classifier and export to ONNX.

## Quick-start

```bash
# 1. Install training dependencies (separate from the service venv)
pip install -r training/requirements.txt

# 2. Prepare your dataset CSV  (see data/sample_labels.csv for schema)
python training/dataset.py --csv data/labels.csv --images data/images/ --out data/splits/

# 3. Fine-tune
python training/train.py --splits data/splits/ --epochs 20 --out checkpoints/

# 4. Evaluate
python training/evaluate.py --splits data/splits/ --checkpoint checkpoints/best.pth

# 5. Export to ONNX
python training/export_onnx.py --checkpoint checkpoints/best.pth --out models/current/
```

## Labels

| Index | Label |
|-------|-------|
| 0 | pigmentation |
| 1 | redness |
| 2 | uneven_texture |
| 3 | dryness |
| 4 | hair_density |
| 5 | other_or_unclear |

## Dataset CSV schema

See `data/sample_labels.csv`. Each row is one image; label columns are binary (0/1).

## Target metrics (pilot baseline)

- 300–500 labelled images for a first run.
- Per-label F1 >= 0.70 before deploying to pilot branch.
- 2 000+ images for a stronger pilot.
