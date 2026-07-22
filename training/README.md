# GenX 360 AI Training & Model Export Pipeline

This directory contains scripts for dataset splitting, fine-tuning, evaluating, and exporting the local ONNX vision model for the GenX 360 CRM microservice.

---

## 🎯 Model Target Architecture

* **Backbone**: MobileNetV3-Large
* **Output Type**: 5-class single-label top prediction with Softmax activation
* **Labels Set (5 classes)**:
  1. `acne`
  2. `dryness`
  3. `hair_thinning`
  4. `pigmentation`
  5. `redness`
* **Export Target**: `models/current/mobilenetv3_large_skin_ai.onnx` & `models/current/labels.json`

---

## 🚀 Execution Workflow

### 1. Dataset Preparation & Splitting
Split an `ImageFolder` dataset structure (`data/Skin_AI_Dataset/<label_name>/*.jpg`) into train, validation, and test splits:

```bash
python training/dataset.py \
    --dataset-dir data/Skin_AI_Dataset/ \
    --out data/splits/
```

### 2. Model Training
Fine-tune MobileNetV3-Large on the 5-label dataset:

```bash
python training/train.py \
    --dataset-dir data/Skin_AI_Dataset/ \
    --splits-dir data/splits/ \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-4 \
    --out checkpoints/
```

### 3. Model Evaluation
Evaluate the fine-tuned checkpoint across test, validation, and train splits to output evaluation metrics:

```bash
python training/evaluate.py \
    --dataset-dir data/Skin_AI_Dataset/ \
    --splits-dir data/splits/ \
    --checkpoint checkpoints/best_mobilenetv3.pth \
    --out scratch/onnx_evaluation_report.json
```

### 4. ONNX Export
Export the PyTorch model checkpoint to ONNX format for local CPU inference in FastAPI:

```bash
python training/export_onnx.py \
    --checkpoint checkpoints/best_mobilenetv3.pth \
    --out models/current/ \
    --version genx-mobilenetv3-v1.0
```

This exports `models/current/mobilenetv3_large_skin_ai.onnx` and `models/current/labels.json`, synced directly with `LocalModelProvider` (`app/local_model_provider.py`).
