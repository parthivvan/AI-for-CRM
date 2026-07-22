"""Unit tests for LocalModelProvider (no real model required)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.local_model_provider import LocalModelProvider, ALL_LABELS
from app.schemas import ImageType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes() -> bytes:
    """Return valid JPEG bytes (224x224 image with noise/gradient)."""
    from PIL import Image
    from io import BytesIO
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# is_ready()
# ---------------------------------------------------------------------------

def test_is_not_ready_when_model_file_absent(tmp_path):
    provider = LocalModelProvider(
        model_path=str(tmp_path / "nonexistent.onnx"),
        labels_path=str(tmp_path / "labels.json"),
    )
    provider.load()
    assert provider.is_ready() is False


def test_is_not_ready_when_labels_file_absent(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"dummy")
    provider = LocalModelProvider(
        model_path=str(tmp_path / "model.onnx"),
        labels_path=str(tmp_path / "labels.json"),
    )
    provider.load()
    assert provider.is_ready() is False


# ---------------------------------------------------------------------------
# Label file parsing
# ---------------------------------------------------------------------------

def test_loads_labels_from_json(tmp_path):
    labels = ALL_LABELS
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))
    model_path = tmp_path / "model.onnx"
    # Provide a fake ONNX file to bypass the path check; session creation will fail
    # but we can verify label loading via a mocked session
    model_path.write_bytes(b"fake-onnx")

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        provider = LocalModelProvider(
            model_path=str(model_path),
            labels_path=str(labels_path),
        )
        provider.load()

    assert provider.is_ready() is True  # session loaded successfully via mock
    assert provider._labels == labels


# ---------------------------------------------------------------------------
# Preprocessing tensor shape
# ---------------------------------------------------------------------------

def test_preprocess_produces_correct_shape():
    image_bytes = _make_jpeg_bytes()
    tensor = LocalModelProvider._preprocess(image_bytes)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------

def test_flags_above_threshold_are_detected(tmp_path):
    labels = ALL_LABELS
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    # High logit for index 0 (acne), others low
    raw_scores = np.array([[5.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_session.run.return_value = [raw_scores]

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        provider = LocalModelProvider(
            model_path=str(model_path),
            labels_path=str(labels_path),
            threshold=0.65,
        )
        provider.load()
        result = provider.analyze(_make_jpeg_bytes(), ImageType.skin)

    assert "acne" in result.detected_flags
    assert len(result.detected_flags) == 1
    assert result.confidence > 0.9

def test_flags_below_threshold_are_not_detected(tmp_path):
    labels = ALL_LABELS
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    # Logits that result in max probability below threshold
    raw_scores = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0]], dtype=np.float32)

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_session.run.return_value = [raw_scores]

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        provider = LocalModelProvider(
            model_path=str(model_path),
            labels_path=str(labels_path),
            threshold=0.65,
        )
        provider.load()
        result = provider.analyze(_make_jpeg_bytes(), ImageType.skin)

    assert len(result.detected_flags) == 0
    # Softmax of 5 equal logits is 1/5 = 0.2
    assert round(result.confidence, 4) == 0.2
