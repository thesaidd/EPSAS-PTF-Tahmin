from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.features import (
    PtfFeatureBuildRequest,
    PtfFeatureBuildSummary,
    PtfFeatureStatusResponse,
)
from ml.features.ptf_features import PtfFeatureService

router = APIRouter(prefix="/api/features", tags=["features"])


def get_ptf_feature_service() -> PtfFeatureService:
    return PtfFeatureService()


@router.get("/ptf/status", response_model=PtfFeatureStatusResponse)
def ptf_feature_status(
    service: PtfFeatureService = Depends(get_ptf_feature_service),
) -> PtfFeatureStatusResponse:
    try:
        return PtfFeatureStatusResponse.model_validate(service.get_status())
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query PTF feature status.",
        ) from exc


@router.post("/ptf/build", response_model=PtfFeatureBuildSummary)
def build_ptf_features(
    request: PtfFeatureBuildRequest,
    service: PtfFeatureService = Depends(get_ptf_feature_service),
) -> PtfFeatureBuildSummary:
    try:
        summary = service.build_and_store_features(
            start_date=request.start_date,
            end_date=request.end_date,
            feature_version=request.feature_version,
        )
        return PtfFeatureBuildSummary.model_validate(summary)
    except (SQLAlchemyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

