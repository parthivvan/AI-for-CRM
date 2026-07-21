"""Unit tests for AIProvider routing: local_model -> gemini_fallback -> deterministic_fallback.

All tests use mocks — no real network calls or ONNX model required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ai_provider import AIProvider
from app.config import Settings
from app.schemas import ImageType, VisionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(**overrides) -> Settings:
    base = {
        "app_env": "test",
        "ai_service_api_key": "test-key",
        "pilot_branch_id": "pilot-branch-001",
        "gemini_api_key": None,
        "enable_gemini_fallback": True,
        "model_threshold": 0.65,
        "model_path": "models/current/model.onnx",
        "model_labels_path": "models/current/labels.json",
        "model_version": "genx-vision-v0.1",
    }
    base.update(overrides)
    return Settings(**base)


def _local_model(ready: bool = True, confidence: float = 0.80) -> MagicMock:
    mock = MagicMock()
    mock.is_ready.return_value = ready
    result = VisionResult(
        detected_flags=["pigmentation"],
        flag_scores={"pigmentation": confidence},
        confidence=confidence,
        provider_used="local_model",
    )
    mock.analyze.return_value = result
    return mock


def _gemini_result() -> VisionResult:
    return VisionResult(
        detected_flags=["redness"],
        flag_scores={},
        confidence=0.75,
        provider_used="gemini_fallback",
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_model_used_when_ready_and_confident():
    settings = _settings()
    local = _local_model(ready=True, confidence=0.80)
    provider = AIProvider(settings, local_model=local)

    with patch.object(provider, "_download_image", new=AsyncMock(return_value=(b"img", "image/jpeg"))):
        result = await provider.analyze_image("https://example.com/img.jpg", ImageType.skin)

    assert result.provider_used == "local_model"
    local.analyze.assert_called_once()


@pytest.mark.asyncio
async def test_deterministic_fallback_when_local_not_ready():
    settings = _settings()
    local = _local_model(ready=False)
    provider = AIProvider(settings, local_model=local)

    result = await provider.analyze_image("https://example.com/pigmentation.jpg", ImageType.skin)

    assert result.provider_used == "deterministic_fallback"
    assert "pigmentation" in result.detected_flags


@pytest.mark.asyncio
async def test_deterministic_fallback_when_local_low_confidence():
    settings = _settings(model_threshold=0.65)
    local = _local_model(ready=True, confidence=0.40)  # below threshold
    provider = AIProvider(settings, local_model=local)

    with patch.object(provider, "_download_image", new=AsyncMock(return_value=(b"img", "image/jpeg"))):
        result = await provider.analyze_image("https://example.com/redness.jpg", ImageType.skin)

    assert result.provider_used == "deterministic_fallback"
    assert "redness" in result.detected_flags


@pytest.mark.asyncio
async def test_model_version_none_for_non_local_provider():
    """model_version should be None when not using local model (tested at services layer)."""
    settings = _settings()
    local = _local_model(ready=False)
    provider = AIProvider(settings, local_model=local)

    result = await provider.analyze_image("https://example.com/img.jpg", ImageType.skin)

    # Services.py sets model_version based on provider_used; here we just confirm
    # deterministic_fallback does not set provider_used to local_model
    assert result.provider_used != "local_model"


# ---------------------------------------------------------------------------
# Invalid branch rejection — POST /analyze
#
# test_api_contracts.py::test_wrong_branch_is_rejected already covers
# POST /recommendations/run; this test covers the same validate_branch()
# logic but via the POST /analyze endpoint.
# ---------------------------------------------------------------------------

def test_invalid_branch_rejected_for_analyze():
    from fastapi.testclient import TestClient
    from app.main import app

    payload = {
        "consultation_id": "consult-001",
        "branch_id": "not-a-real-branch",
        "client_id": "client-001",
        "image_url": "https://example.com/skin.jpg",
        "image_type": "skin",
    }

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            json=payload,
            headers={"X-API-Key": "dev-ai-service-key"},
        )

    assert response.status_code == 400
    assert "pilot branch" in response.json()["detail"].lower()
