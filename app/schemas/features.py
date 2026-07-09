from datetime import date, datetime

from pydantic import BaseModel, Field, model_validator


class PtfFeatureBuildRequest(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    feature_version: str = Field(default="v1", min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_date_range(self) -> "PtfFeatureBuildRequest":
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.end_date < self.start_date
        ):
            raise ValueError("end_date must be on or after start_date")
        return self


class PtfFeatureBuildSummary(BaseModel):
    source_rows: int
    feature_rows_built: int
    feature_rows_inserted_or_updated: int
    min_timestamp: str | None
    max_timestamp: str | None
    feature_version: str
    warnings: list[str]
    errors: list[str]


class PtfFeatureStatusResponse(BaseModel):
    total_rows: int
    min_timestamp: datetime | None
    max_timestamp: datetime | None
    latest_updated_at: datetime | None
    feature_versions: list[str]

