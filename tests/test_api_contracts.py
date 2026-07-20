from fastapi.testclient import TestClient

from app.main import app



def test_root_contract():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["health"] == "/health"


def test_uppercase_health_contract():
    with TestClient(app) as client:
        response = client.get("/HEALTH")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
def test_health_contract():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_contract_with_sample_crm_data():
    payload = {
        "consultation_id": "consult-001",
        "branch_id": "pilot-branch-001",
        "client_id": "client-001",
        "image_url": "https://example.com/pigmentation-redness.jpg",
        "image_type": "skin",
    }

    with TestClient(app) as client:
        response = client.post("/analyze", json=payload, headers={"X-API-Key": "dev-ai-service-key"})

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_id"].startswith("analysis-")
    assert "pigmentation" in body["detected_flags"]
    assert body["recommended_treatments"]
    assert body["requires_human_review"] is True
    # Transparency fields
    assert body["provider_used"] in {"local_model", "gemini_fallback", "deterministic_fallback"}
    assert isinstance(body["flag_scores"], dict)
    # model_version is set only for local_model; may be None or absent otherwise
    assert "model_version" in body


def test_recommendations_contract_with_sample_crm_data():
    payload = {"branch_id": "pilot-branch-001", "mode": "manual"}

    with TestClient(app) as client:
        response = client.post("/recommendations/run", json=payload, headers={"X-API-Key": "dev-ai-service-key"})

    assert response.status_code == 200
    body = response.json()
    assert body["branch_id"] == "pilot-branch-001"
    assert body["generated_count"] == 1
    assert body["recommendations"][0]["score"] >= 60


def test_wrong_branch_is_rejected():
    payload = {"branch_id": "other-branch", "mode": "manual"}

    with TestClient(app) as client:
        response = client.post("/recommendations/run", json=payload, headers={"X-API-Key": "dev-ai-service-key"})

    assert response.status_code == 400


