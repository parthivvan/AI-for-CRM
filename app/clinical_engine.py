"""Clinical Decision Engine for GenX 360 CRM.

Fuses multi-source Evidence objects (Vision AI, Questionnaire, CRM History)
with configurable clinical rules (data/clinical_rules.json) to produce
structured, explainable treatment recommendations.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.schemas import ClientIntakeQuestionnaire, EvidenceItem

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "clinical_rules.json"


@lru_cache
def load_clinical_rules() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "weights": {
            "vision_weight": 0.45,
            "questionnaire_weight": 0.30,
            "history_weight": 0.20,
            "clinic_priority_weight": 0.05,
        },
        "contraindications": {"is_pregnant": ["laser", "peel", "retinoid", "prp"]},
        "boosts": {"sensitive_skin_calming": 25, "questionnaire_concern_match": 30, "history_concern_match": 20},
        "penalties": {"sensitive_skin_exfoliation": -15},
        "confidence_thresholds": {"high": 75, "medium": 45},
    }


def evaluate_clinical_decision_engine(
    visual_flags: list[str],
    flag_scores: dict[str, float],
    treatment_catalog: list[dict[str, Any]],
    questionnaire: ClientIntakeQuestionnaire | None = None,
    client_history: dict[str, Any] | None = None,
    max_results: int = 3,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Evaluate multi-source Evidence objects and produce structured explainable recommendations.

    Returns:
        tuple[list[dict], list[str]]: (explainable_recommendations, contraindication_warnings)
    """
    config = load_clinical_rules()
    contra_rules = config.get("contraindications", {})
    boosts = config.get("boosts", {})
    penalties = config.get("penalties", {})
    thresholds = config.get("confidence_thresholds", {"high": 75, "medium": 45})

    contraindication_warnings: list[str] = []
    scored_treatments: list[dict[str, Any]] = []

    is_pregnant = questionnaire.is_pregnant if questionnaire else False
    skin_type = questionnaire.skin_type if questionnaire else None
    patient_concerns = questionnaire.primary_concerns if questionnaire else []
    recent_concerns = client_history.get("recent_concerns", []) if client_history else []

    # Check active pregnancy contraindications
    pregnant_forbidden = contra_rules.get("is_pregnant", ["laser", "peel", "retinoid", "prp"])

    for treatment in treatment_catalog:
        t_id = treatment.get("treatment_id", "")
        t_name = treatment.get("name", "")
        name_lower = t_name.lower()
        keywords = {str(k).lower() for k in treatment.get("keywords", [])}
        
        treatment_contraindications: list[str] = []

        # 1. Contraindications Filter
        if is_pregnant and any(kw in keywords or kw in name_lower for kw in pregnant_forbidden):
            warning = f"Contraindication: '{t_name}' filtered out due to active pregnancy status."
            contraindication_warnings.append(warning)
            treatment_contraindications.append("Active pregnancy status")
            continue  # Omit contraindicated treatment from shortlist

        evidence_list: list[EvidenceItem] = []
        why_recommended: list[str] = []
        score = 0

        # 2. Vision AI Evidence Objects
        for flag in visual_flags:
            flag_prob = flag_scores.get(flag, 0.5)
            if flag in keywords or any(flag in kw for kw in keywords):
                score += int(40 * flag_prob)
                evidence_list.append(
                    EvidenceItem(
                        source="Vision AI",
                        finding=flag.capitalize().replace("_", " "),
                        confidence=round(flag_prob, 2),
                        explanation=f"Detected visible {flag} with {int(flag_prob * 100)}% confidence",
                    )
                )
                why_recommended.append(f"Matches detected {flag.replace('_', ' ')} condition")

        # 3. Patient Questionnaire Evidence Objects
        for concern in patient_concerns:
            c_lower = concern.lower()
            if c_lower in keywords or any(c_lower in kw for kw in keywords):
                score += boosts.get("questionnaire_concern_match", 30)
                evidence_list.append(
                    EvidenceItem(
                        source="Questionnaire",
                        finding=f"Primary concern: {concern}",
                        explanation=f"Patient self-reported '{concern}' in intake form",
                    )
                )
                why_recommended.append(f"Directly addresses patient concern: '{concern}'")

        if skin_type == "sensitive":
            if any(kw in keywords or kw in name_lower for kw in ["calming", "soothing", "barrier", "hydration"]):
                score += boosts.get("sensitive_skin_calming", 25)
                evidence_list.append(
                    EvidenceItem(
                        source="Questionnaire",
                        finding="Sensitive skin type",
                        explanation="Formulated for sensitive skin barrier support",
                    )
                )
                why_recommended.append("Safe & supportive for sensitive skin")
            elif any(kw in keywords or kw in name_lower for kw in ["peel", "acid", "exfoliation"]):
                score += penalties.get("sensitive_skin_exfoliation", -15)
                evidence_list.append(
                    EvidenceItem(
                        source="Questionnaire",
                        finding="Sensitive skin caution",
                        explanation="May require patch test due to active exfoliation ingredients",
                    )
                )

        # 4. CRM History Evidence Objects
        for hist_concern in recent_concerns:
            h_lower = str(hist_concern).lower()
            if h_lower in keywords or any(h_lower in kw for kw in keywords):
                score += boosts.get("history_concern_match", 20)
                evidence_list.append(
                    EvidenceItem(
                        source="CRM History",
                        finding=f"Past concern: {hist_concern}",
                        explanation=f"Recorded in client consultation history",
                    )
                )
                why_recommended.append(f"Aligns with past consultation history ('{hist_concern}')")

        if score > 0:
            final_score = min(score, 100)
            if final_score >= thresholds.get("high", 75):
                confidence_str = "High"
            elif final_score >= thresholds.get("medium", 45):
                confidence_str = "Medium"
            else:
                confidence_str = "Low"

            summary_reasons = [e.explanation for e in evidence_list if e.explanation]

            scored_treatments.append({
                "treatment_id": t_id,
                "name": t_name,
                "score": final_score,
                "confidence": confidence_str,
                "evidence": [e.model_dump() for e in evidence_list],
                "contraindications": treatment_contraindications,
                "why_recommended": list(dict.fromkeys(why_recommended)),
                "reason": "; ".join(dict.fromkeys(summary_reasons)),
            })

    # Rank by score descending
    scored_treatments.sort(key=lambda item: (-item["score"], item["name"]))

    ranked_shortlist = [
        {
            "treatment_id": item["treatment_id"],
            "name": item["name"],
            "rank": index + 1,
            "score": item["score"],
            "confidence": item["confidence"],
            "evidence": item["evidence"],
            "contraindications": item["contraindications"],
            "why_recommended": item["why_recommended"],
            "reason": item["reason"],
        }
        for index, item in enumerate(scored_treatments[:max_results])
    ]

    return ranked_shortlist, list(dict.fromkeys(contraindication_warnings))
