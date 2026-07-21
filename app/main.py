import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse

from app.ai_provider import AIProvider
from app.config import Settings, get_settings
from app.crm_client import CRMClient
from app.local_model_provider import LocalModelProvider
from app.scheduler import configure_scheduler
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    OutcomeRequest,
    RecommendationRunRequest,
    RecommendationRunResponse,
)
from app.security import require_api_key
from app.services import run_image_analysis, run_recommendations
from app.storage import create_storage_engine, init_db, save_outcome

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = get_settings()
    app.state.engine = create_storage_engine(app.state.settings)
    init_db(app.state.engine)
    app.state.crm_client = CRMClient(app.state.settings)
    # Load local ONNX model (safe no-op if file is absent)
    local_model = LocalModelProvider(
        model_path=app.state.settings.model_path,
        labels_path=app.state.settings.model_labels_path,
        threshold=app.state.settings.model_threshold,
    )
    local_model.load()
    app.state.local_model = local_model
    app.state.ai_provider = AIProvider(app.state.settings, local_model=local_model)
    app.state.scheduler = configure_scheduler(
        app.state.settings,
        app.state.crm_client,
        app.state.ai_provider,
        app.state.engine,
    )
    yield
    if app.state.scheduler:
        app.state.scheduler.shutdown(wait=False)


app = FastAPI(
    title="GenX 360 AI Service",
    version="0.1.0",
    description="Pilot AI service for GenX 360 CRM image analysis and upsell recommendations.",
    lifespan=lifespan,
)


def app_settings(request: Request) -> Settings:
    return request.app.state.settings

@app.get("/test-ui", response_class=HTMLResponse, include_in_schema=False)
def test_ui():
    with open("test_ui.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/image", include_in_schema=False)
def get_local_image(path: str):
    import os
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    
    if not os.path.exists(path) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "GenX 360 AI Service",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /analyze",
            "POST /recommendations/run",
            "POST /outcomes",
        ],
    }

@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(app_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        pilot_branch_id=settings.pilot_branch_id,
        scheduler_enabled=settings.enable_scheduler,
    )


@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
async def analyze(request_body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    return await run_image_analysis(
        request_body,
        request.app.state.settings,
        request.app.state.crm_client,
        request.app.state.ai_provider,
        request.app.state.engine,
    )


@app.post("/recommendations/run", response_model=RecommendationRunResponse, dependencies=[Depends(require_api_key)])
async def recommendations_run(request_body: RecommendationRunRequest, request: Request) -> RecommendationRunResponse:
    return await run_recommendations(
        request_body,
        request.app.state.settings,
        request.app.state.crm_client,
        request.app.state.ai_provider,
        request.app.state.engine,
    )


@app.post("/outcomes", dependencies=[Depends(require_api_key)])
def outcomes(request_body: OutcomeRequest, request: Request) -> dict[str, str]:
    save_outcome(request.app.state.engine, request_body)
    return {"status": "recorded"}

@app.get("/HEALTH", response_model=HealthResponse, include_in_schema=False)
def health_uppercase(settings: Settings = Depends(app_settings)) -> HealthResponse:
    return health(settings)

