from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class ImageType(str, Enum):
    skin = "skin"
    scalp = "scalp"
    hair = "hair"
    body = "body"


class ClientIntakeQuestionnaire(BaseModel):
    skin_type: Literal["oily", "dry", "combination", "sensitive", "normal"] | None = None
    primary_concerns: list[str] = Field(default_factory=list)
    is_pregnant: bool = False
    allergies: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    recent_treatments: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    consultation_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    image_url: str
    image_type: ImageType
    intake_questionnaire: ClientIntakeQuestionnaire | None = None


class EvidenceItem(BaseModel):
    source: str
    finding: str
    confidence: float | None = None
    explanation: str | None = None


class TreatmentRecommendation(BaseModel):
    treatment_id: str
    name: str
    rank: int = Field(ge=1)
    score: int = Field(default=80, ge=0, le=100)
    confidence: str = "Medium"
    evidence: list[EvidenceItem] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    why_recommended: list[str] = Field(default_factory=list)
    reason: str = ""


class AnalyzeResponse(BaseModel):
    analysis_id: str
    detected_flags: list[str]
    flag_scores: dict[str, float] = Field(default_factory=dict)
    recommended_treatments: list[TreatmentRecommendation]
    consultant_brief: str
    confidence: float = Field(ge=0, le=1)
    provider_used: str = "deterministic_fallback"
    model_version: str | None = None
    requires_human_review: bool = True


class RecommendationRunRequest(BaseModel):
    branch_id: str = Field(min_length=1)
    mode: Literal["manual", "scheduled"] = "manual"


class UpsellRecommendation(BaseModel):
    client_id: str
    score: int = Field(ge=0, le=100)
    reason: str
    suggested_offer: str
    draft_message: str


class RecommendationRunResponse(BaseModel):
    branch_id: str
    generated_count: int
    recommendations: list[UpsellRecommendation]


class OutcomeRequest(BaseModel):
    object_type: Literal["analysis", "recommendation"]
    object_id: str
    branch_id: str
    staff_action: Literal["accepted", "edited", "ignored", "rejected"]
    final_outcome: Literal["booked_consultation", "purchased_package", "no_response", "not_suitable"] | None = None
    notes: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_env: str
    pilot_branch_id: str
    scheduler_enabled: bool


class VisionResult(BaseModel):
    detected_flags: list[str]
    flag_scores: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    provider_used: str = "deterministic_fallback"
    raw: dict[str, Any] = Field(default_factory=dict)


