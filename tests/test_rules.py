from app.rules import map_indicators_to_treatments, score_upsell_client, should_recommend


def test_maps_detected_flags_to_ranked_treatments():
    catalog = [
        {
            "treatment_id": "tx-1",
            "name": "Laser Pigmentation Correction",
            "keywords": ["laser", "pigmentation"],
        },
        {
            "treatment_id": "tx-2",
            "name": "Calming Facial",
            "keywords": ["facial", "redness"],
        },
    ]

    result = map_indicators_to_treatments(["pigmentation", "redness"], catalog)

    assert [item["treatment_id"] for item in result] == ["tx-2", "tx-1"]
    assert result[0]["rank"] == 1


def test_scores_high_intent_upsell_client():
    client = {
        "client_id": "client-001",
        "sessions_total": 6,
        "sessions_used": 5,
        "days_since_last_visit": 40,
        "missed_follow_up": True,
        "purchase_count": 3,
        "recent_concerns": ["pigmentation"],
        "available_offers": ["Laser Pigmentation Correction"],
    }

    result = score_upsell_client(client)

    assert result["score"] == 100
    assert should_recommend(result["score"])
    assert result["suggested_offer"] == "Laser Pigmentation Correction"

