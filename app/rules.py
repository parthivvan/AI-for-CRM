import json
from functools import lru_cache
from pathlib import Path
from typing import Any

RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "treatment_rules.json"


@lru_cache
def load_rules() -> dict[str, Any]:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def map_indicators_to_treatments(
    detected_flags: list[str],
    treatment_catalog: list[dict[str, Any]],
    max_results: int = 3,
) -> list[dict[str, Any]]:
    rules = load_rules()["image_rules"]
    scored: list[dict[str, Any]] = []

    for treatment in treatment_catalog:
        keywords = {str(k).lower() for k in treatment.get("keywords", [])}
        name = str(treatment.get("name", "")).lower()
        score = 0
        reasons: list[str] = []

        for flag in detected_flags:
            rule = rules.get(flag)
            if not rule:
                continue
            rule_keywords = {str(k).lower() for k in rule.get("treatment_keywords", [])}
            if keywords.intersection(rule_keywords) or any(keyword in name for keyword in rule_keywords):
                score += 1
                reasons.append(rule["reason"])

        if score:
            scored.append(
                {
                    "treatment_id": treatment["treatment_id"],
                    "name": treatment["name"],
                    "score": score,
                    "reason": " ".join(dict.fromkeys(reasons)),
                }
            )

    scored.sort(key=lambda item: (-item["score"], item["name"]))
    return [
        {
            "treatment_id": item["treatment_id"],
            "name": item["name"],
            "rank": index + 1,
            "reason": item["reason"],
        }
        for index, item in enumerate(scored[:max_results])
    ]


def score_upsell_client(client: dict[str, Any]) -> dict[str, Any]:
    score = 0
    signals: list[str] = []

    sessions_total = int(client.get("sessions_total") or 0)
    sessions_used = int(client.get("sessions_used") or 0)
    days_since_last_visit = int(client.get("days_since_last_visit") or 0)
    purchase_count = int(client.get("purchase_count") or 0)
    recent_concerns = client.get("recent_concerns") or []

    if sessions_total > 0 and sessions_used / sessions_total >= 0.8:
        score += 35
        signals.append("package is almost consumed")

    if client.get("missed_follow_up"):
        score += 20
        signals.append("missed a follow-up")

    if days_since_last_visit >= 30:
        score += 15
        signals.append("has not visited recently")

    if purchase_count >= 2:
        score += 15
        signals.append("has a strong purchase history")

    if recent_concerns:
        score += 15
        signals.append(f"recent concern match: {', '.join(map(str, recent_concerns))}")

    score = min(score, 100)
    available_offers = client.get("available_offers") or []
    suggested_offer = available_offers[0] if available_offers else "Follow-up consultation"

    return {
        "client_id": client["client_id"],
        "score": score,
        "signals": signals,
        "suggested_offer": suggested_offer,
    }


def should_recommend(score: int) -> bool:
    return score >= int(load_rules().get("upsell_threshold", 60))

