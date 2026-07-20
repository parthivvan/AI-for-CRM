import logging
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy.engine import Engine

from app.ai_provider import AIProvider
from app.config import Settings
from app.crm_client import CRMClient
from app.rules import map_indicators_to_treatments, score_upsell_client, should_recommend
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    RecommendationRunRequest,
    RecommendationRunResponse,
    TreatmentRecommendation,
    UpsellRecommendation,
)
from app.storage import save_analysis, save_recommendations, serialize_for_crm

logger = logging.getLogger(__name__)


def validate_branch(branch_id: str, settings: Settings) -> None:
    if settings.pilot_branch_id and branch_id != settings.pilot_branch_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Branch {branch_id!r} is outside the configured pilot branch.",
        )


async def run_image_analysis(
    request: AnalyzeRequest,
    settings: Settings,
    crm_client: CRMClient,
    ai_provider: AIProvider,
    engine: Engine,
) -> AnalyzeResponse:
    validate_branch(request.branch_id, settings)
    try:
        treatment_catalog = await crm_client.get_treatments(request.branch_id)
    except Exception as exc:
        logger.exception("Failed to load treatment catalog")
        raise HTTPException(status_code=502, detail="Unable to load treatment catalog from CRM.") from exc

    vision = await ai_provider.analyze_image(str(request.image_url), request.image_type)
    mapped = map_indicators_to_treatments(vision.detected_flags, treatment_catalog)
    brief = await ai_provider.generate_consultant_brief(vision.detected_flags, mapped, request.image_type)
    model_version = settings.model_version if vision.provider_used == "local_model" else None
    response = AnalyzeResponse(
        analysis_id=f"analysis-{uuid4().hex}",
        detected_flags=vision.detected_flags,
        flag_scores=vision.flag_scores,
        recommended_treatments=[TreatmentRecommendation(**item) for item in mapped],
        consultant_brief=brief,
        confidence=vision.confidence,
        provider_used=vision.provider_used,
        model_version=model_version,
        requires_human_review=True,
    )
    save_analysis(engine, request, response)

    if settings.push_analysis_to_crm:
        try:
            await crm_client.post_analysis(request.consultation_id, serialize_for_crm(response))
        except Exception:
            logger.exception("CRM analysis push failed for consultation_id=%s", request.consultation_id)

    return response


async def run_recommendations(
    request: RecommendationRunRequest,
    settings: Settings,
    crm_client: CRMClient,
    ai_provider: AIProvider,
    engine: Engine,
) -> RecommendationRunResponse:
    validate_branch(request.branch_id, settings)
    try:
        clients = await crm_client.get_clients(request.branch_id)
    except Exception as exc:
        logger.exception("Failed to load clients")
        raise HTTPException(status_code=502, detail="Unable to load clients from CRM.") from exc

    recommendations: list[UpsellRecommendation] = []
    for client in clients:
        score_result = score_upsell_client(client)
        if not should_recommend(score_result["score"]):
            continue
        generated = await ai_provider.generate_upsell_text(client, score_result)
        recommendations.append(
            UpsellRecommendation(
                client_id=score_result["client_id"],
                score=score_result["score"],
                reason=generated["reason"],
                suggested_offer=score_result["suggested_offer"],
                draft_message=generated["draft_message"],
            )
        )

    response = RecommendationRunResponse(
        branch_id=request.branch_id,
        generated_count=len(recommendations),
        recommendations=recommendations,
    )
    save_recommendations(engine, request.branch_id, recommendations)

    try:
        await crm_client.post_recommendations(serialize_for_crm(response))
    except Exception:
        logger.exception("CRM recommendation push failed for branch_id=%s", request.branch_id)

    return response

