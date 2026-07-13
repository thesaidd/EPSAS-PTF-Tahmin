from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    version: str
    environment: str


class SystemReadinessResponse(BaseModel):
    api_healthy: bool
    db_reachable: bool
    ptf_rows: int | None = None
    feature_rows: int | None = None
    latest_forecast_decision_run: str | None = None
    latest_day_ahead_forecast_run: str | None = None
    latest_pipeline_run: str | None = None
    latest_monitoring_status: str | None = None
    dashboard_url: str
    swagger_url: str
    mlflow_url: str
    details: dict[str, Any] = Field(default_factory=dict)
