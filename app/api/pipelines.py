from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.pipelines import (
    DailyForecastPipelineRunRequest,
    PipelineRunsResponse,
    PipelineRunSummary,
)
from ml.pipelines.daily_forecast_pipeline import DailyForecastPipelineService

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


def get_daily_forecast_pipeline_service() -> DailyForecastPipelineService:
    return DailyForecastPipelineService()


@router.post(
    "/daily-forecast/run",
    response_model=PipelineRunSummary,
)
def run_daily_forecast_pipeline(
    request: DailyForecastPipelineRunRequest,
    service: DailyForecastPipelineService = Depends(
        get_daily_forecast_pipeline_service
    ),
) -> PipelineRunSummary:
    try:
        return PipelineRunSummary.model_validate(
            service.run_pipeline(
                target_date=request.target_date,
                ingest_start_date=request.ingest_start_date,
                ingest_end_date=request.ingest_end_date,
                skip_ingestion=request.skip_ingestion,
                skip_feature_build=request.skip_feature_build,
            )
        )
    except (SQLAlchemyError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get(
    "/daily-forecast/status",
    response_model=PipelineRunSummary | dict[str, object],
)
def daily_forecast_pipeline_status(
    service: DailyForecastPipelineService = Depends(
        get_daily_forecast_pipeline_service
    ),
) -> PipelineRunSummary | dict[str, object]:
    try:
        latest = service.get_latest_pipeline_status()
        return PipelineRunSummary.model_validate(latest) if latest else {}
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query latest daily forecast pipeline status.",
        ) from exc


@router.get(
    "/daily-forecast/runs",
    response_model=PipelineRunsResponse,
)
def daily_forecast_pipeline_runs(
    limit: int = Query(default=20, ge=1, le=100),
    service: DailyForecastPipelineService = Depends(
        get_daily_forecast_pipeline_service
    ),
) -> PipelineRunsResponse:
    try:
        return PipelineRunsResponse(
            runs=[
                PipelineRunSummary.model_validate(row)
                for row in service.get_pipeline_runs(limit=limit)
            ]
        )
    except (SQLAlchemyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
