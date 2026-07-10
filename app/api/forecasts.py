from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.forecasts import (
    DayAheadForecastGenerateRequest,
    DayAheadForecastGenerateResponse,
    DayAheadForecastLatestResponse,
    DayAheadForecastStatusResponse,
)
from ml.inference.day_ahead_ptf import DayAheadPtfForecastService

router = APIRouter(prefix="/api/forecasts", tags=["forecasts"])


def get_day_ahead_ptf_service() -> DayAheadPtfForecastService:
    return DayAheadPtfForecastService()


@router.post(
    "/ptf/day-ahead/generate",
    response_model=DayAheadForecastGenerateResponse,
)
def generate_day_ahead_ptf_forecast(
    request: DayAheadForecastGenerateRequest,
    service: DayAheadPtfForecastService = Depends(get_day_ahead_ptf_service),
) -> DayAheadForecastGenerateResponse:
    try:
        summary = service.run_day_ahead_forecast(
            target_date=request.target_date,
            horizon_hours=request.horizon_hours,
            model_version=request.model_version,
        )
        return DayAheadForecastGenerateResponse.model_validate(summary)
    except (RuntimeError, SQLAlchemyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get(
    "/ptf/day-ahead/latest",
    response_model=DayAheadForecastLatestResponse,
)
def latest_day_ahead_ptf_forecast(
    service: DayAheadPtfForecastService = Depends(get_day_ahead_ptf_service),
) -> DayAheadForecastLatestResponse:
    try:
        return DayAheadForecastLatestResponse.model_validate(
            service.get_latest_forecast()
        )
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query latest day-ahead forecast.",
        ) from exc


@router.get(
    "/ptf/day-ahead/status",
    response_model=DayAheadForecastStatusResponse,
)
def day_ahead_ptf_forecast_status(
    service: DayAheadPtfForecastService = Depends(get_day_ahead_ptf_service),
) -> DayAheadForecastStatusResponse:
    try:
        return DayAheadForecastStatusResponse.model_validate(service.get_status())
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query day-ahead forecast status.",
        ) from exc
