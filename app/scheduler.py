import asyncio
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.ai_provider import AIProvider
from app.config import Settings
from app.crm_client import CRMClient
from app.schemas import RecommendationRunRequest
from app.services import run_recommendations

logger = logging.getLogger(__name__)


def configure_scheduler(settings: Settings, crm_client: CRMClient, ai_provider: AIProvider, engine) -> BackgroundScheduler | None:
    if not settings.enable_scheduler:
        return None

    scheduler = BackgroundScheduler(timezone="UTC")

    def job() -> None:
        logger.info("Starting scheduled recommendation run for branch_id=%s", settings.pilot_branch_id)
        asyncio.run(
            run_recommendations(
                RecommendationRunRequest(branch_id=settings.pilot_branch_id, mode="scheduled"),
                settings,
                crm_client,
                ai_provider,
                engine,
            )
        )

    scheduler.add_job(job, "cron", hour=settings.recommendation_cron_hour, id="daily_recommendations", replace_existing=True)
    scheduler.start()
    return scheduler

