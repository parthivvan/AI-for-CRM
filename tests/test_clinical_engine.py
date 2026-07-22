import pytest
from app.clinical_engine import evaluate_clinical_decision_engine
from app.schemas import ClientIntakeQuestionnaire

CATALOG = [
    {
        "treatment_id": "tx-laser-pigmentation",
        "name": "Laser Pigmentation Correction",
        "keywords": ["laser", "pigmentation", "brightening"],
    },
    {
        "treatment_id": "tx-chemical-peel",
        "name": "Advanced Chemical Peel",
        "keywords": ["peel", "texture", "brightening"],
    },
    {
        "treatment_id": "tx-calming-facial",
        "name": "Calming Barrier Facial",
        "keywords": ["soothing", "facial", "calming", "redness", "hydration", "barrier"],
    },
]


def test_explainable_recommendations_and_evidence_objects():
    questionnaire = ClientIntakeQuestionnaire(
        skin_type="dry",
        primary_concerns=["pigmentation"],
        is_pregnant=False,
    )
    history = {"recent_concerns": ["pigmentation"]}

    shortlist, warnings = evaluate_clinical_decision_engine(
        visual_flags=["pigmentation"],
        flag_scores={"pigmentation": 0.88},
        treatment_catalog=CATALOG,
        questionnaire=questionnaire,
        client_history=history,
    )

    assert len(shortlist) > 0
    top = shortlist[0]
    assert top["name"] == "Laser Pigmentation Correction"
    assert top["score"] >= 75
    assert top["confidence"] == "High"
    
    # Verify structured evidence list
    sources = [e["source"] for e in top["evidence"]]
    assert "Vision AI" in sources
    assert "Questionnaire" in sources
    assert "CRM History" in sources

    # Verify structured why_recommended
    assert any("Matches detected pigmentation" in w for w in top["why_recommended"])
    assert len(warnings) == 0


def test_pregnancy_contraindication_filters_laser_and_peel():
    questionnaire = ClientIntakeQuestionnaire(
        skin_type="normal",
        primary_concerns=["pigmentation"],
        is_pregnant=True,  # Active pregnancy
    )

    shortlist, warnings = evaluate_clinical_decision_engine(
        visual_flags=["pigmentation", "redness"],
        flag_scores={"pigmentation": 0.85, "redness": 0.70},
        treatment_catalog=CATALOG,
        questionnaire=questionnaire,
    )

    # Laser and Chemical Peel should be filtered out due to pregnancy
    treatment_names = [t["name"] for t in shortlist]
    assert "Laser Pigmentation Correction" not in treatment_names
    assert "Advanced Chemical Peel" not in treatment_names
    assert "Calming Barrier Facial" in treatment_names

    # Explicit contraindication warnings returned
    assert len(warnings) >= 2
    assert any("active pregnancy" in w for w in warnings)


def test_sensitive_skin_boosts_calming_facial():
    questionnaire = ClientIntakeQuestionnaire(
        skin_type="sensitive",
        primary_concerns=["redness"],
        is_pregnant=False,
    )

    shortlist, warnings = evaluate_clinical_decision_engine(
        visual_flags=["redness"],
        flag_scores={"redness": 0.90},
        treatment_catalog=CATALOG,
        questionnaire=questionnaire,
    )

    assert shortlist[0]["name"] == "Calming Barrier Facial"
    assert any("sensitive skin" in w.lower() for w in shortlist[0]["why_recommended"])
