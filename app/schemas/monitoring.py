from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MonitoringSnapshotRequest(BaseModel):
    max_ptf_age_hours: int = 168
    expected_forecast_horizon_hours: int = 24

    @model_validator(mode="after")
    def validate_request(self) -> "MonitoringSnapshotRequest":
        if self.max_ptf_age_hours <= 0:
            raise ValueError("max_ptf_age_hours must be positive")
        if self.expected_forecast_horizon_hours <= 0:
            raise ValueError("expected_forecast_horizon_hours must be positive")
        return self


class MonitoringSnapshotResponse(BaseModel):
    snapshot_id: str
    status: str
    created_at: datetime | str
    data_freshness: dict[str, Any] = Field(default_factory=dict)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    pipeline_health: dict[str, Any] = Field(default_factory=dict)
    forecast_health: dict[str, Any] = Field(default_factory=dict)
    model_quality: dict[str, Any] = Field(default_factory=dict)
    uncertainty_quality: dict[str, Any] = Field(default_factory=dict)
    risk_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class MonitoringCompactStatusResponse(BaseModel):
    status: str | None = None
    created_at: datetime | str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    latest_pipeline_status: str | None = None
    latest_forecast_run_id: str | None = None
    latest_data_timestamp: datetime | str | None = None
    latest_model_metrics: dict[str, Any] = Field(default_factory=dict)


class MonitoringSnapshotsResponse(BaseModel):
    snapshots: list[MonitoringSnapshotResponse]
