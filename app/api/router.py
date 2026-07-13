from fastapi import APIRouter, Depends

from app.api.epias import router as epias_router
from app.api.features import router as features_router
from app.api.forecasts import router as forecasts_router
from app.api.monitoring import router as monitoring_router
from app.api.models import router as models_router
from app.api.pipelines import router as pipelines_router
from app.core.config import settings
from app.schemas.system import (
    HealthResponse,
    SystemReadinessResponse,
    VersionResponse,
)
from app.services.readiness import SystemReadinessService

router = APIRouter()
router.include_router(epias_router)
router.include_router(features_router)
router.include_router(forecasts_router)
router.include_router(monitoring_router)
router.include_router(models_router)
router.include_router(pipelines_router)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.get("/version", response_model=VersionResponse, tags=["system"])
def version() -> VersionResponse:
    return VersionResponse(
        version=settings.app_version,
        environment=settings.environment,
    )


def get_system_readiness_service() -> SystemReadinessService:
    return SystemReadinessService()


@router.get(
    "/api/system/readiness",
    response_model=SystemReadinessResponse,
    tags=["system"],
)
def system_readiness(
    service: SystemReadinessService = Depends(get_system_readiness_service),
) -> SystemReadinessResponse:
    return SystemReadinessResponse.model_validate(service.get_readiness())
