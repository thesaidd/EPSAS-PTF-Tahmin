from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.monitoring import (
    MonitoringCompactStatusResponse,
    MonitoringSnapshotRequest,
    MonitoringSnapshotResponse,
    MonitoringSnapshotsResponse,
)
from ml.monitoring.ptf_monitoring import PtfMonitoringService

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


def get_ptf_monitoring_service() -> PtfMonitoringService:
    return PtfMonitoringService()


@router.post("/ptf/snapshot", response_model=MonitoringSnapshotResponse)
def create_ptf_monitoring_snapshot(
    request: MonitoringSnapshotRequest,
    service: PtfMonitoringService = Depends(get_ptf_monitoring_service),
) -> MonitoringSnapshotResponse:
    try:
        return MonitoringSnapshotResponse.model_validate(
            service.build_snapshot(
                max_ptf_age_hours=request.max_ptf_age_hours,
                expected_forecast_horizon_hours=(
                    request.expected_forecast_horizon_hours
                ),
            )
        )
    except (SQLAlchemyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get(
    "/ptf/latest",
    response_model=MonitoringSnapshotResponse | dict[str, object],
)
def latest_ptf_monitoring_snapshot(
    service: PtfMonitoringService = Depends(get_ptf_monitoring_service),
) -> MonitoringSnapshotResponse | dict[str, object]:
    try:
        latest = service.get_latest_snapshot()
        return MonitoringSnapshotResponse.model_validate(latest) if latest else {}
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query latest monitoring snapshot.",
        ) from exc


@router.get(
    "/ptf/status",
    response_model=MonitoringCompactStatusResponse | dict[str, object],
)
def ptf_monitoring_status(
    service: PtfMonitoringService = Depends(get_ptf_monitoring_service),
) -> MonitoringCompactStatusResponse | dict[str, object]:
    try:
        compact = service.get_compact_status()
        return MonitoringCompactStatusResponse.model_validate(compact) if compact else {}
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query monitoring status.",
        ) from exc


@router.get("/ptf/snapshots", response_model=MonitoringSnapshotsResponse)
def list_ptf_monitoring_snapshots(
    limit: int = Query(default=20, ge=1, le=100),
    service: PtfMonitoringService = Depends(get_ptf_monitoring_service),
) -> MonitoringSnapshotsResponse:
    try:
        return MonitoringSnapshotsResponse(
            snapshots=[
                MonitoringSnapshotResponse.model_validate(row)
                for row in service.list_snapshots(limit=limit)
            ]
        )
    except (SQLAlchemyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
