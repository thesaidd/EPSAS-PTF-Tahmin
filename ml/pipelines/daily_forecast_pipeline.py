import json
import logging
import uuid
from collections.abc import Callable
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from data_pipeline.epias.ptf_ingestion import PtfIngestionService
from ml.features.ptf_features import PtfFeatureService
from ml.inference.day_ahead_ptf import DayAheadPtfForecastService

logger = logging.getLogger(__name__)

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")
PIPELINE_NAME = "daily_forecast"
STATUS_RUNNING = "RUNNING"
STATUS_SUCCESS = "SUCCESS"
STATUS_PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"


class DailyForecastPipelineService:
    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
        ptf_ingestion_service: PtfIngestionService | None = None,
        feature_service: PtfFeatureService | None = None,
        day_ahead_service: DayAheadPtfForecastService | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.ptf_ingestion_service = ptf_ingestion_service or PtfIngestionService()
        self.feature_service = feature_service or PtfFeatureService()
        self.day_ahead_service = day_ahead_service or DayAheadPtfForecastService()

    def run_pipeline(
        self,
        target_date: date | None = None,
        ingest_start_date: date | None = None,
        ingest_end_date: date | None = None,
        skip_ingestion: bool = False,
        skip_feature_build: bool = False,
    ) -> dict[str, Any]:
        resolved_target_date = target_date or _today_istanbul() + timedelta(days=1)
        resolved_ingest_end = ingest_end_date or _today_istanbul()
        resolved_ingest_start = ingest_start_date or (
            resolved_ingest_end - timedelta(days=3)
        )
        if resolved_ingest_end < resolved_ingest_start:
            raise ValueError("ingest_end_date must be on or after ingest_start_date")

        pipeline_run_id = str(uuid.uuid4())
        started_at = datetime.now(tz=ISTANBUL_TIMEZONE)
        steps: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        errors: list[str] = []
        forecast_run_id: str | None = None

        self.store_pipeline_run_start(
            pipeline_run_id=pipeline_run_id,
            target_date=resolved_target_date,
            ingest_start_date=resolved_ingest_start,
            ingest_end_date=resolved_ingest_end,
            started_at=started_at,
        )

        if skip_ingestion:
            steps["ptf_ingestion"] = {
                "status": STATUS_SKIPPED,
                "message": "PTF ingestion skipped by request.",
            }
        else:
            steps["ptf_ingestion"] = self.run_ptf_ingestion_step(
                resolved_ingest_start,
                resolved_ingest_end,
            )
            warnings.extend(steps["ptf_ingestion"].get("warnings", []))
            errors.extend(steps["ptf_ingestion"].get("errors", []))

        should_continue_after_ingestion = (
            steps.get("ptf_ingestion", {}).get("status")
            in {STATUS_SUCCESS, STATUS_SKIPPED}
            or self.has_existing_ptf_data()
        )
        if not should_continue_after_ingestion:
            final_status = STATUS_FAILED
            errors.append("PTF ingestion failed and no existing ptf_hourly data is available.")
            finished_at = datetime.now(tz=ISTANBUL_TIMEZONE)
            self.update_pipeline_run_finish(
                pipeline_run_id,
                final_status,
                finished_at,
                forecast_run_id,
                steps,
                warnings,
                errors,
            )
            return self._summary(
                pipeline_run_id,
                final_status,
                started_at,
                finished_at,
                resolved_target_date,
                resolved_ingest_start,
                resolved_ingest_end,
                forecast_run_id,
                steps,
                warnings,
                errors,
            )

        if skip_feature_build:
            steps["feature_build"] = {
                "status": STATUS_SKIPPED,
                "message": "Feature build skipped by request.",
            }
        else:
            steps["feature_build"] = self.run_feature_build_step(
                resolved_ingest_start,
                resolved_ingest_end,
            )
            warnings.extend(steps["feature_build"].get("warnings", []))
            errors.extend(steps["feature_build"].get("errors", []))

        steps["day_ahead_forecast"] = self.run_day_ahead_forecast_step(
            resolved_target_date
        )
        warnings.extend(steps["day_ahead_forecast"].get("warnings", []))
        errors.extend(steps["day_ahead_forecast"].get("errors", []))
        forecast_run_id = steps["day_ahead_forecast"].get("forecast_run_id")

        final_status = self.aggregate_status(steps)
        finished_at = datetime.now(tz=ISTANBUL_TIMEZONE)
        self.update_pipeline_run_finish(
            pipeline_run_id,
            final_status,
            finished_at,
            forecast_run_id,
            steps,
            warnings,
            errors,
        )
        return self._summary(
            pipeline_run_id,
            final_status,
            started_at,
            finished_at,
            resolved_target_date,
            resolved_ingest_start,
            resolved_ingest_end,
            forecast_run_id,
            steps,
            warnings,
            errors,
        )

    def run_ptf_ingestion_step(
        self,
        ingest_start_date: date,
        ingest_end_date: date,
    ) -> dict[str, Any]:
        try:
            summary = self.ptf_ingestion_service.ingest_ptf_range(
                ingest_start_date,
                ingest_end_date,
                chunk_days=3,
            )
            errors = list(summary.get("errors") or [])
            return {
                "status": STATUS_PARTIAL_SUCCESS if errors else STATUS_SUCCESS,
                "records_fetched": int(summary.get("records_fetched") or 0),
                "records_inserted_or_updated": int(
                    summary.get("records_inserted_or_updated") or 0
                ),
                "chunks_processed": int(summary.get("chunks_processed") or 0),
                "missing_hours": list(summary.get("missing_hours") or []),
                "warnings": [],
                "errors": errors,
            }
        except Exception as exc:
            logger.exception("Daily pipeline PTF ingestion step failed.")
            return {
                "status": STATUS_FAILED,
                "warnings": [
                    "PTF ingestion failed; continuing with existing data if available."
                ],
                "errors": [str(exc)],
            }

    def run_feature_build_step(
        self,
        ingest_start_date: date,
        ingest_end_date: date,
    ) -> dict[str, Any]:
        try:
            summary = self.feature_service.build_and_store_features(
                start_date=ingest_start_date,
                end_date=ingest_end_date,
                feature_version="v1",
            )
            errors = list(summary.get("errors") or [])
            return {
                "status": STATUS_PARTIAL_SUCCESS if errors else STATUS_SUCCESS,
                "source_rows": int(summary.get("source_rows") or 0),
                "feature_rows_built": int(summary.get("feature_rows_built") or 0),
                "feature_rows_inserted_or_updated": int(
                    summary.get("feature_rows_inserted_or_updated") or 0
                ),
                "warnings": list(summary.get("warnings") or []),
                "errors": errors,
            }
        except Exception as exc:
            logger.exception("Daily pipeline feature build step failed.")
            return {
                "status": STATUS_FAILED,
                "warnings": [],
                "errors": [str(exc)],
            }

    def run_day_ahead_forecast_step(self, target_date: date) -> dict[str, Any]:
        try:
            summary = self.day_ahead_service.run_day_ahead_forecast(
                target_date=target_date,
                horizon_hours=24,
                model_version="day_ahead_v1",
            )
            errors = list(summary.get("errors") or [])
            status = STATUS_FAILED if errors else STATUS_SUCCESS
            return {
                "status": status,
                "forecast_run_id": summary.get("forecast_run_id"),
                "rows_generated": int(summary.get("rows_generated") or 0),
                "target_date": summary.get("target_date"),
                "selected_model": summary.get("selected_model"),
                "warnings": list(summary.get("warnings") or []),
                "errors": errors,
            }
        except Exception as exc:
            logger.exception("Daily pipeline day-ahead forecast step failed.")
            return {
                "status": STATUS_FAILED,
                "warnings": [],
                "errors": [str(exc)],
            }

    def aggregate_status(self, steps: dict[str, dict[str, Any]]) -> str:
        forecast_status = steps.get("day_ahead_forecast", {}).get("status")
        if forecast_status != STATUS_SUCCESS:
            return STATUS_FAILED
        non_success = [
            step
            for step in steps.values()
            if step.get("status") not in {STATUS_SUCCESS, STATUS_SKIPPED}
        ]
        return STATUS_PARTIAL_SUCCESS if non_success else STATUS_SUCCESS

    def has_existing_ptf_data(self) -> bool:
        try:
            status = self.ptf_ingestion_service.get_status()
            return int(status.get("total_rows") or 0) > 0
        except Exception:
            return False

    def store_pipeline_run_start(
        self,
        pipeline_run_id: str,
        target_date: date,
        ingest_start_date: date,
        ingest_end_date: date,
        started_at: datetime,
    ) -> None:
        statement = text(
            """
            INSERT INTO pipeline_runs (
                pipeline_run_id,
                pipeline_name,
                status,
                started_at,
                target_date,
                ingest_start_date,
                ingest_end_date,
                steps,
                warnings,
                errors
            )
            VALUES (
                :pipeline_run_id,
                :pipeline_name,
                :status,
                :started_at,
                :target_date,
                :ingest_start_date,
                :ingest_end_date,
                CAST(:steps AS JSONB),
                CAST(:warnings AS JSONB),
                CAST(:errors AS JSONB)
            )
            """
        )
        with self.session_factory() as session:
            session.execute(
                statement,
                {
                    "pipeline_run_id": pipeline_run_id,
                    "pipeline_name": PIPELINE_NAME,
                    "status": STATUS_RUNNING,
                    "started_at": started_at,
                    "target_date": target_date,
                    "ingest_start_date": ingest_start_date,
                    "ingest_end_date": ingest_end_date,
                    "steps": "{}",
                    "warnings": "[]",
                    "errors": "[]",
                },
            )
            session.commit()

    def update_pipeline_run_finish(
        self,
        pipeline_run_id: str,
        status: str,
        finished_at: datetime,
        forecast_run_id: str | None,
        steps: dict[str, Any],
        warnings: list[str],
        errors: list[str],
    ) -> None:
        statement = text(
            """
            UPDATE pipeline_runs
            SET
                status = :status,
                finished_at = :finished_at,
                forecast_run_id = :forecast_run_id,
                steps = CAST(:steps AS JSONB),
                warnings = CAST(:warnings AS JSONB),
                errors = CAST(:errors AS JSONB)
            WHERE pipeline_run_id = :pipeline_run_id
            """
        )
        with self.session_factory() as session:
            session.execute(
                statement,
                {
                    "pipeline_run_id": pipeline_run_id,
                    "status": status,
                    "finished_at": finished_at,
                    "forecast_run_id": forecast_run_id,
                    "steps": json.dumps(_json_ready(steps)),
                    "warnings": json.dumps(_json_ready(warnings)),
                    "errors": json.dumps(_json_ready(errors)),
                },
            )
            session.commit()

    def get_latest_pipeline_status(self) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT *
                    FROM pipeline_runs
                    WHERE pipeline_name = :pipeline_name
                    ORDER BY started_at DESC, id DESC
                    LIMIT 1
                    """
                ),
                {"pipeline_name": PIPELINE_NAME},
            ).mappings().one_or_none()
        return _pipeline_row_to_dict(row) if row is not None else {}

    def get_pipeline_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self.session_factory() as session:
            rows = [
                _pipeline_row_to_dict(row)
                for row in session.execute(
                    text(
                        """
                        SELECT *
                        FROM pipeline_runs
                        WHERE pipeline_name = :pipeline_name
                        ORDER BY started_at DESC, id DESC
                        LIMIT :limit
                        """
                    ),
                    {"pipeline_name": PIPELINE_NAME, "limit": limit},
                ).mappings()
            ]
        return rows

    def _summary(
        self,
        pipeline_run_id: str,
        status: str,
        started_at: datetime,
        finished_at: datetime | None,
        target_date: date,
        ingest_start_date: date,
        ingest_end_date: date,
        forecast_run_id: str | None,
        steps: dict[str, Any],
        warnings: list[str],
        errors: list[str],
    ) -> dict[str, Any]:
        return {
            "pipeline_run_id": pipeline_run_id,
            "pipeline_name": PIPELINE_NAME,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "target_date": target_date,
            "ingest_start_date": ingest_start_date,
            "ingest_end_date": ingest_end_date,
            "forecast_run_id": forecast_run_id,
            "steps": _json_ready(steps),
            "warnings": warnings,
            "errors": errors,
        }


def _today_istanbul() -> date:
    return datetime.now(tz=ISTANBUL_TIMEZONE).date()


def _pipeline_row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return _json_ready(dict(row))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value
