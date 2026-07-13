from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class DailyForecastPipelineRunRequest(BaseModel):
    target_date: date | None = None
    ingest_start_date: date | None = None
    ingest_end_date: date | None = None
    skip_ingestion: bool = False
    skip_feature_build: bool = False

    @model_validator(mode="after")
    def validate_date_range(self) -> "DailyForecastPipelineRunRequest":
        if (
            self.ingest_start_date is not None
            and self.ingest_end_date is not None
            and self.ingest_end_date < self.ingest_start_date
        ):
            raise ValueError("ingest_end_date must be on or after ingest_start_date")
        return self


class PipelineRunSummary(BaseModel):
    pipeline_run_id: str
    pipeline_name: str
    status: str
    started_at: datetime | str | None = None
    finished_at: datetime | str | None = None
    target_date: date | str | None = None
    ingest_start_date: date | str | None = None
    ingest_end_date: date | str | None = None
    forecast_run_id: str | None = None
    steps: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PipelineRunsResponse(BaseModel):
    runs: list[PipelineRunSummary]
