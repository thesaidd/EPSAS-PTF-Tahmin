from fastapi import APIRouter

from app.api.epias import router as epias_router
from app.core.config import settings
from app.schemas.system import HealthResponse, VersionResponse

router = APIRouter()
router.include_router(epias_router)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.get("/version", response_model=VersionResponse, tags=["system"])
def version() -> VersionResponse:
    return VersionResponse(
        version=settings.app_version,
        environment=settings.environment,
    )
