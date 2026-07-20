# models/

This directory holds the trained ONNX vision model used by the GenX 360 AI service.

## Structure

```
models/
  current/
    model.onnx       ← active ONNX model (not committed to git)
    labels.json      ← label list matching model output order
    .gitkeep         ← keeps the directory in git
```

## Dropping in a trained model

1. Export your fine-tuned classifier with `training/export_onnx.py`.
2. Copy the resulting `model.onnx` to `models/current/model.onnx`.
3. Restart the FastAPI server — it will load automatically on startup.
4. If the file is absent, the service falls back to Gemini → deterministic path.

## Label order

Labels in `labels.json` **must** match the output neuron order of your ONNX model:

```json
["pigmentation","redness","uneven_texture","dryness","hair_density","other_or_unclear"]
```
