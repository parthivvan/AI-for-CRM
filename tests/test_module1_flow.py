"""Integration tests for Module 1 end-to-end flow (Analysis -> Shortlist -> Outcomes)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ImageType


def test_module1_end_to_end_flow_and_outcome_logging():
    client = TestClient(app)

    # 1. Post image analysis request
    analyze_payload = {
        "consultation_id": "consult-module1-test",
        "branch_id": "pilot-branch-001",
        "client_id": "client-module1-test",
        "image_url": "https://example.com/skin_pigmentation.jpg",
        "image_type": ImageType.skin,
    }

    res_analyze = client.post(
        "/analyze",
        json=analyze_payload,
        headers={"X-API-Key": "dev-ai-service-key"},
    )

    assert res_analyze.status_code == 200
    data = res_analyze.json()

    assert data["analysis_id"].startswith("analysis-")
    assert "detected_flags" in data
    assert "recommended_treatments" in data
    assert len(data["recommended_treatments"]) > 0
    assert "consultant_brief" in data
    assert len(data["consultant_brief"]) > 0
    assert data["requires_human_review"] is True

    # Check Softmax scores sum to ~1.0 if flag_scores is present
    flag_scores = data.get("flag_scores", {})
    if flag_scores:
        total_prob = sum(flag_scores.values())
        assert abs(total_prob - 1.0) < 1e-3

    # 2. Log consultant outcome feedback
    outcome_payload = {
        "object_type": "analysis",
        "object_id": data["analysis_id"],
        "branch_id": "pilot-branch-001",
        "staff_action": "accepted",
        "final_outcome": "booked_consultation",
        "notes": "Consultant accepted the treatment shortlist.",
    }

    res_outcome = client.post(
        "/outcomes",
        json=outcome_payload,
        headers={"X-API-Key": "dev-ai-service-key"},
    )

    assert res_outcome.status_code == 200
    assert res_outcome.json() == {"status": "recorded"}
