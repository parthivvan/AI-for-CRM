"""Local ONNX model provider for GenX 360 vision analysis.

Loads a fine-tuned multi-label classifier exported as ONNX.
Falls back gracefully when the model file is not yet present.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from app.schemas import ImageType, VisionResult

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)

# Labels the model is trained to predict (order must match model output)
ALL_LABELS: list[str] = [
    "pigmentation",
    "redness",
    "uneven_texture",
    "dryness",
    "hair_density",
    "other_or_unclear",
]

# Labels excluded from the routing confidence check
_ROUTING_EXCLUDED = {"other_or_unclear"}

# Image pre-processing constants (ImageNet-style normalisation)
_INPUT_SIZE = (224, 224)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class LocalModelProvider:
    """Wraps an ONNX multi-label classifier for image analysis."""

    def __init__(self, model_path: str, labels_path: str, threshold: float = 0.65) -> None:
        self._model_path = Path(model_path)
        self._labels_path = Path(labels_path)
        self._threshold = threshold
        self._session: "ort.InferenceSession | None" = None
        self._labels: list[str] = []
        self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load ONNX model and labels. Logs a warning if files are missing."""
        if not self._model_path.exists():
            logger.warning(
                "LocalModelProvider: model file not found at %s — falling back to next provider.",
                self._model_path,
            )
            return

        if not self._labels_path.exists():
            logger.warning(
                "LocalModelProvider: labels file not found at %s — falling back to next provider.",
                self._labels_path,
            )
            return

        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            logger.warning("onnxruntime is not installed — local model provider unavailable.")
            return

        try:
            self._labels = json.loads(self._labels_path.read_text(encoding="utf-8"))
            self._session = ort.InferenceSession(
                str(self._model_path),
                providers=["CPUExecutionProvider"],
            )
            self._ready = True
            logger.info(
                "LocalModelProvider: loaded model from %s with %d labels.",
                self._model_path,
                len(self._labels),
            )
        except Exception:
            logger.exception("LocalModelProvider: failed to load model — falling back to next provider.")

    def is_ready(self) -> bool:
        """Return True only when the ONNX session is loaded and usable."""
        return self._ready

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def analyze(self, image_bytes: bytes, image_type: ImageType) -> VisionResult:
        """Run inference and return a VisionResult."""
        if not self._ready or self._session is None:
            raise RuntimeError("LocalModelProvider.analyze() called while not ready.")

        tensor = self._preprocess(image_bytes)
        input_name = self._session.get_inputs()[0].name
        raw_output: list[Any] = self._session.run(None, {input_name: tensor})
        scores_array = np.array(raw_output[0][0], dtype=np.float32)

        labels = self._labels if self._labels else ALL_LABELS
        flag_scores: dict[str, float] = {
            label: float(np.clip(scores_array[i], 0.0, 1.0))
            for i, label in enumerate(labels)
            if i < len(scores_array)
        }

        # Detected flags: any label at or above threshold
        detected_flags = [label for label, score in flag_scores.items() if score >= self._threshold]

        # Routing confidence: max score excluding other_or_unclear
        routing_scores = {k: v for k, v in flag_scores.items() if k not in _ROUTING_EXCLUDED}
        max_confidence = max(routing_scores.values()) if routing_scores else 0.0

        return VisionResult(
            detected_flags=detected_flags,
            flag_scores=flag_scores,
            confidence=round(max_confidence, 4),
            provider_used="local_model",
            raw={"model_path": str(self._model_path)},
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(image_bytes: bytes) -> np.ndarray:
        """Convert raw image bytes to a normalised float32 CHW tensor (1, 3, 224, 224)."""
        from io import BytesIO

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = img.resize(_INPUT_SIZE, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0, 1]
        arr = arr.transpose(2, 0, 1)  # CHW
        arr = (arr - _MEAN) / _STD
        return arr[np.newaxis, ...]  # (1, 3, 224, 224)
