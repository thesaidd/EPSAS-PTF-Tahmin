from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class DayAheadForecastGenerateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    target_date: date | None = None
    horizon_hours: int = 24
    model_version: str = "day_ahead_v1"

    @model_validator(mode="after")
    def validate_request(self) -> "DayAheadForecastGenerateRequest":
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")
        return self


class DayAheadForecastGenerateResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    forecast_run_id: str
    target_date: str | None
    horizon_hours: int
    selected_model: str
    xgboost_training_run_id: str | None
    gpr_run_id: str | None
    decision_run_id: str | None
    generation_method: str
    model_version: str
    rows_generated: int
    min_timestamp: str | None
    max_timestamp: str | None
    warnings: list[str]
    errors: list[str]


class DayAheadForecastRow(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int | None = None
    forecast_run_id: str
    target_date: date | str
    timestamp: datetime | str
    horizon_hour: int
    selected_model: str
    xgboost_prediction: float
    residual_mean: float | None = None
    residual_std: float | None = None
    forecast_ptf: float
    lower_bound_95: float | None = None
    upper_bound_95: float | None = None
    interval_width_95: float | None = None
    risk_level: str | None = None
    xgboost_training_run_id: str | None = None
    gpr_run_id: str | None = None
    decision_run_id: str | None = None
    model_version: str
    generation_method: str
    warnings: list[str] | dict[str, Any] | None = None
    generated_at: datetime | str | None = None
    created_at: datetime | str | None = None


class DayAheadForecastLatestResponse(BaseModel):
    latest_forecast_run_id: str | None
    target_date: str | None
    generated_at: str | None
    rows: int
    summary: dict[str, Any]
    forecasts: list[DayAheadForecastRow]


class DayAheadForecastStatusResponse(BaseModel):
    total_rows: int
    total_runs: int
    latest_forecast_run_id: str | None
    latest_target_date: date | str | None
    latest_generated_at: datetime | str | None
    available_model_versions: list[str]
