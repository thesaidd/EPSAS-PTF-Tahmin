import json
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import SessionLocal

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")
STATUS_HEALTHY = "HEALTHY"
STATUS_WARNING = "WARNING"
STATUS_CRITICAL = "CRITICAL"
DEFAULT_MAX_PTF_AGE_HOURS = 168
DEFAULT_EXPECTED_FORECAST_HORIZON_HOURS = 24
DATA_QUALITY_WINDOW_DAYS = 30
MODEL_R2_HEALTHY_THRESHOLD = 0.75
MODEL_R2_WARNING_THRESHOLD = 0.50
MODEL_MAE_WARNING_THRESHOLD = 750.0
MODEL_MAE_CRITICAL_THRESHOLD = 1250.0
UNCERTAINTY_COVERAGE_HEALTHY_MIN = 85.0
UNCERTAINTY_COVERAGE_HEALTHY_MAX = 98.0
UNCERTAINTY_COVERAGE_WARNING_MIN = 70.0
RISK_WARNING_HIGH_HOURS = 8
RISK_CRITICAL_HIGH_HOURS = 16


class PtfMonitoringService:
    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self.session_factory = session_factory

    def build_snapshot(
        self,
        max_ptf_age_hours: int = DEFAULT_MAX_PTF_AGE_HOURS,
        expected_forecast_horizon_hours: int = DEFAULT_EXPECTED_FORECAST_HORIZON_HOURS,
        store: bool = True,
    ) -> dict[str, Any]:
        if max_ptf_age_hours <= 0:
            raise ValueError("max_ptf_age_hours must be positive")
        if expected_forecast_horizon_hours <= 0:
            raise ValueError("expected_forecast_horizon_hours must be positive")

        data_freshness = self.check_data_freshness(max_ptf_age_hours)
        data_quality = self.check_data_quality()
        pipeline_health = self.check_pipeline_health()
        forecast_health = self.check_forecast_health(
            expected_forecast_horizon_hours
        )
        model_quality = self.check_model_quality()
        uncertainty_quality = self.check_uncertainty_quality()
        risk_summary = self.check_risk_summary()
        sections = [
            data_freshness,
            data_quality,
            pipeline_health,
            forecast_health,
            model_quality,
            uncertainty_quality,
            risk_summary,
        ]
        warnings = _collect_messages(sections, "warnings")
        errors = _collect_messages(sections, "errors")
        snapshot = {
            "snapshot_id": str(uuid.uuid4()),
            "status": self.determine_overall_status(sections),
            "created_at": datetime.now(tz=ISTANBUL_TIMEZONE),
            "data_freshness": data_freshness,
            "data_quality": data_quality,
            "pipeline_health": pipeline_health,
            "forecast_health": forecast_health,
            "model_quality": model_quality,
            "uncertainty_quality": uncertainty_quality,
            "risk_summary": risk_summary,
            "warnings": warnings,
            "errors": errors,
        }
        if store:
            self.store_snapshot(snapshot)
        return _json_ready(snapshot)

    def check_data_freshness(
        self,
        max_ptf_age_hours: int = DEFAULT_MAX_PTF_AGE_HOURS,
    ) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS total_rows,
                        MIN("timestamp") AS min_timestamp,
                        MAX("timestamp") AS max_timestamp
                    FROM ptf_hourly
                    """
                )
            ).mappings().one()

        total_rows = int(row["total_rows"] or 0)
        max_timestamp = row["max_timestamp"]
        if total_rows == 0 or max_timestamp is None:
            return _section(
                STATUS_CRITICAL,
                {
                    "total_rows": total_rows,
                    "min_timestamp": row["min_timestamp"],
                    "max_timestamp": max_timestamp,
                    "latest_ptf_age_hours": None,
                },
                errors=["No PTF data exists in ptf_hourly."],
            )

        now_utc = datetime.now(tz=ZoneInfo("UTC"))
        latest_age_hours = (
            now_utc - _as_aware_datetime(max_timestamp).astimezone(ZoneInfo("UTC"))
        ).total_seconds() / 3600
        status = (
            STATUS_HEALTHY
            if latest_age_hours <= max_ptf_age_hours
            else STATUS_WARNING
        )
        warnings = []
        if status == STATUS_WARNING:
            warnings.append(
                f"Latest PTF timestamp is stale: {latest_age_hours:.1f} hours old."
            )
        return _section(
            status,
            {
                "total_rows": total_rows,
                "min_timestamp": row["min_timestamp"],
                "max_timestamp": max_timestamp,
                "latest_ptf_age_hours": latest_age_hours,
                "max_ptf_age_hours": max_ptf_age_hours,
            },
            warnings=warnings,
        )

    def check_data_quality(self) -> dict[str, Any]:
        with self.session_factory() as session:
            bounds = session.execute(
                text(
                    """
                    SELECT MAX("timestamp") AS max_timestamp
                    FROM ptf_hourly
                    """
                )
            ).mappings().one()
            max_timestamp = bounds["max_timestamp"]
            if max_timestamp is None:
                return _section(
                    STATUS_CRITICAL,
                    {
                        "window_days": DATA_QUALITY_WINDOW_DAYS,
                        "missing_hour_count": None,
                        "duplicate_timestamp_count": None,
                        "negative_ptf_count": None,
                        "null_ptf_count": None,
                    },
                    errors=["Cannot evaluate data quality because ptf_hourly is empty."],
                )

            window_end = _as_aware_datetime(max_timestamp)
            window_start = window_end - timedelta(days=DATA_QUALITY_WINDOW_DAYS)
            quality = session.execute(
                text(
                    """
                    WITH window_rows AS (
                        SELECT "timestamp", ptf_tl
                        FROM ptf_hourly
                        WHERE "timestamp" >= :window_start
                          AND "timestamp" <= :window_end
                    ),
                    duplicate_rows AS (
                        SELECT COUNT(*) AS duplicate_count
                        FROM (
                            SELECT "timestamp"
                            FROM window_rows
                            GROUP BY "timestamp"
                            HAVING COUNT(*) > 1
                        ) AS duplicates
                    )
                    SELECT
                        (SELECT COUNT(*) FROM window_rows) AS actual_hours,
                        (SELECT duplicate_count FROM duplicate_rows)
                            AS duplicate_timestamp_count,
                        COUNT(*) FILTER (WHERE ptf_tl < 0) AS negative_ptf_count,
                        COUNT(*) FILTER (WHERE ptf_tl IS NULL) AS null_ptf_count
                    FROM window_rows
                    """
                ),
                {"window_start": window_start, "window_end": window_end},
            ).mappings().one()

        expected_hours = int(
            ((window_end.replace(minute=0, second=0, microsecond=0)
              - window_start.replace(minute=0, second=0, microsecond=0))
             .total_seconds() // 3600)
            + 1
        )
        actual_hours = int(quality["actual_hours"] or 0)
        duplicate_count = int(quality["duplicate_timestamp_count"] or 0)
        negative_count = int(quality["negative_ptf_count"] or 0)
        null_count = int(quality["null_ptf_count"] or 0)
        missing_count = max(expected_hours - actual_hours, 0)
        return self.evaluate_data_quality(
            expected_hours=expected_hours,
            actual_hours=actual_hours,
            missing_count=missing_count,
            duplicate_count=duplicate_count,
            negative_count=negative_count,
            null_count=null_count,
            window_start=window_start,
            window_end=window_end,
        )

    def check_pipeline_health(self) -> dict[str, Any]:
        with self.session_factory() as session:
            latest = session.execute(
                text(
                    """
                    SELECT *
                    FROM pipeline_runs
                    WHERE pipeline_name = 'daily_forecast'
                    ORDER BY started_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).mappings().one_or_none()
            counts = session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'SUCCESS') AS success_count,
                        COUNT(*) FILTER (WHERE status = 'FAILED') AS failure_count
                    FROM (
                        SELECT status
                        FROM pipeline_runs
                        WHERE pipeline_name = 'daily_forecast'
                        ORDER BY started_at DESC, id DESC
                        LIMIT 10
                    ) AS recent
                    """
                )
            ).mappings().one()

        if latest is None:
            return _section(
                STATUS_CRITICAL,
                {
                    "latest_pipeline_run_id": None,
                    "recent_success_count": 0,
                    "recent_failure_count": 0,
                },
                errors=["No daily forecast pipeline run exists."],
            )
        latest_status = latest["status"]
        status = {
            "SUCCESS": STATUS_HEALTHY,
            "PARTIAL_SUCCESS": STATUS_WARNING,
            "FAILED": STATUS_CRITICAL,
            "RUNNING": STATUS_WARNING,
        }.get(str(latest_status), STATUS_WARNING)
        messages = []
        if status == STATUS_WARNING:
            messages.append(f"Latest pipeline status is {latest_status}.")
        errors = []
        if status == STATUS_CRITICAL:
            errors.append(f"Latest pipeline status is {latest_status}.")
        return _section(
            status,
            {
                "latest_pipeline_run_id": latest["pipeline_run_id"],
                "latest_status": latest_status,
                "latest_started_at": latest["started_at"],
                "latest_finished_at": latest["finished_at"],
                "latest_target_date": latest["target_date"],
                "latest_forecast_run_id": latest["forecast_run_id"],
                "recent_success_count": int(counts["success_count"] or 0),
                "recent_failure_count": int(counts["failure_count"] or 0),
            },
            warnings=messages,
            errors=errors,
        )

    def check_forecast_health(
        self,
        expected_horizon_hours: int = DEFAULT_EXPECTED_FORECAST_HORIZON_HOURS,
    ) -> dict[str, Any]:
        with self.session_factory() as session:
            latest_run_id = session.scalar(
                text(
                    """
                    SELECT forecast_run_id
                    FROM day_ahead_forecasts
                    ORDER BY generated_at DESC, id DESC
                    LIMIT 1
                    """
                )
            )
            if latest_run_id is None:
                return _section(
                    STATUS_CRITICAL,
                    {
                        "latest_forecast_run_id": None,
                        "latest_rows": 0,
                        "expected_horizon_hours": expected_horizon_hours,
                    },
                    errors=["No day-ahead forecast has been generated."],
                )
            row = session.execute(
                text(
                    """
                    SELECT
                        forecast_run_id,
                        MIN(target_date) AS target_date,
                        MAX(generated_at) AS generated_at,
                        COUNT(*) AS rows,
                        COUNT(DISTINCT horizon_hour) AS distinct_horizon_hours
                    FROM day_ahead_forecasts
                    WHERE forecast_run_id = :forecast_run_id
                    GROUP BY forecast_run_id
                    """
                ),
                {"forecast_run_id": latest_run_id},
            ).mappings().one()

        return self.evaluate_forecast_health(
            forecast_run_id=row["forecast_run_id"],
            target_date=row["target_date"],
            generated_at=row["generated_at"],
            rows=int(row["rows"] or 0),
            distinct_horizon_hours=int(row["distinct_horizon_hours"] or 0),
            expected_horizon_hours=expected_horizon_hours,
        )

    def check_model_quality(self) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT
                        selected_model,
                        mae,
                        rmse,
                        r2,
                        smape,
                        gpr_comparison,
                        created_at
                    FROM forecast_decision_metrics
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).mappings().one_or_none()

        if row is None or row["r2"] is None:
            return _section(
                STATUS_CRITICAL,
                {"latest_selected_model": None},
                errors=["No forecast decision model quality metrics are available."],
            )
        r2 = float(row["r2"])
        mae = _optional_float(row["mae"])
        status = self.evaluate_model_quality_status(r2, mae)
        warnings: list[str] = []
        errors: list[str] = []
        if status == STATUS_WARNING:
            warnings.append("Latest model quality is below healthy thresholds.")
        if status == STATUS_CRITICAL:
            errors.append("Latest model quality is below critical thresholds.")
        comparison = row["gpr_comparison"] or {}
        return _section(
            status,
            {
                "latest_selected_model": row["selected_model"],
                "mae": mae,
                "rmse": _optional_float(row["rmse"]),
                "r2": r2,
                "smape": _optional_float(row["smape"]),
                "selected_vs_gpr_improvement_pct": comparison.get(
                    "selected_vs_gpr_improvement_pct"
                )
                if isinstance(comparison, dict)
                else None,
                "created_at": row["created_at"],
            },
            warnings=warnings,
            errors=errors,
        )

    def check_uncertainty_quality(self) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT
                        interval_coverage_95,
                        mean_interval_width,
                        low_risk_count,
                        medium_risk_count,
                        high_risk_count,
                        created_at
                    FROM forecast_decision_metrics
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).mappings().one_or_none()
        if row is None or row["interval_coverage_95"] is None:
            return _section(
                STATUS_CRITICAL,
                {"interval_coverage_95": None},
                errors=["No uncertainty quality metrics are available."],
            )
        coverage = float(row["interval_coverage_95"])
        status = self.evaluate_uncertainty_status(coverage)
        warnings: list[str] = []
        errors: list[str] = []
        if status == STATUS_WARNING:
            warnings.append("95% interval coverage is outside the healthy range.")
        if status == STATUS_CRITICAL:
            errors.append("95% interval coverage is critically low or unavailable.")
        return _section(
            status,
            {
                "interval_coverage_95": coverage,
                "mean_interval_width": _optional_float(row["mean_interval_width"]),
                "low_risk_count": int(row["low_risk_count"] or 0),
                "medium_risk_count": int(row["medium_risk_count"] or 0),
                "high_risk_count": int(row["high_risk_count"] or 0),
                "created_at": row["created_at"],
            },
            warnings=warnings,
            errors=errors,
        )

    def check_risk_summary(self) -> dict[str, Any]:
        with self.session_factory() as session:
            latest_run_id = session.scalar(
                text(
                    """
                    SELECT forecast_run_id
                    FROM day_ahead_forecasts
                    ORDER BY generated_at DESC, id DESC
                    LIMIT 1
                    """
                )
            )
            if latest_run_id is None:
                return _section(
                    STATUS_CRITICAL,
                    {"latest_forecast_run_id": None},
                    errors=["No day-ahead forecast risk data is available."],
                )
            rows = [
                dict(row)
                for row in session.execute(
                    text(
                        """
                        SELECT forecast_ptf, interval_width_95, risk_level
                        FROM day_ahead_forecasts
                        WHERE forecast_run_id = :forecast_run_id
                        """
                    ),
                    {"forecast_run_id": latest_run_id},
                ).mappings()
            ]
        return self.evaluate_risk_summary(latest_run_id, rows)

    def determine_overall_status(self, sections: list[dict[str, Any]]) -> str:
        statuses = [section.get("status") for section in sections]
        if STATUS_CRITICAL in statuses:
            return STATUS_CRITICAL
        if STATUS_WARNING in statuses:
            return STATUS_WARNING
        return STATUS_HEALTHY

    def evaluate_data_quality(
        self,
        expected_hours: int,
        actual_hours: int,
        missing_count: int,
        duplicate_count: int,
        negative_count: int,
        null_count: int,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> dict[str, Any]:
        warnings: list[str] = []
        errors: list[str] = []
        status = STATUS_HEALTHY
        if missing_count > 0:
            status = STATUS_WARNING
            warnings.append(f"{missing_count} hourly PTF rows are missing.")
        if duplicate_count or null_count:
            status = STATUS_CRITICAL
            errors.append("Duplicate timestamps or null PTF values were found.")
        if negative_count:
            status = STATUS_WARNING if status == STATUS_HEALTHY else status
            warnings.append(f"{negative_count} negative PTF values were found.")
        return _section(
            status,
            {
                "window_days": DATA_QUALITY_WINDOW_DAYS,
                "window_start": window_start,
                "window_end": window_end,
                "expected_hours": expected_hours,
                "actual_hours": actual_hours,
                "missing_hour_count": missing_count,
                "duplicate_timestamp_count": duplicate_count,
                "negative_ptf_count": negative_count,
                "null_ptf_count": null_count,
            },
            warnings=warnings,
            errors=errors,
        )

    def evaluate_forecast_health(
        self,
        forecast_run_id: str,
        target_date: Any,
        generated_at: Any,
        rows: int,
        distinct_horizon_hours: int,
        expected_horizon_hours: int,
    ) -> dict[str, Any]:
        missing = max(expected_horizon_hours - rows, 0)
        status = STATUS_HEALTHY
        warnings: list[str] = []
        if rows < expected_horizon_hours:
            status = STATUS_WARNING
            warnings.append(
                f"Latest forecast has {rows}/{expected_horizon_hours} rows."
            )
        return _section(
            status,
            {
                "latest_forecast_run_id": forecast_run_id,
                "latest_target_date": target_date,
                "latest_generated_at": generated_at,
                "latest_rows": rows,
                "expected_horizon_hours": expected_horizon_hours,
                "missing_horizon_hours": missing,
                "distinct_horizon_hours": distinct_horizon_hours,
            },
            warnings=warnings,
        )

    def evaluate_model_quality_status(
        self,
        r2: float | None,
        mae: float | None,
    ) -> str:
        if r2 is None:
            return STATUS_CRITICAL
        if r2 < MODEL_R2_WARNING_THRESHOLD:
            return STATUS_CRITICAL
        if mae is not None and mae >= MODEL_MAE_CRITICAL_THRESHOLD:
            return STATUS_CRITICAL
        if r2 < MODEL_R2_HEALTHY_THRESHOLD:
            return STATUS_WARNING
        if mae is not None and mae >= MODEL_MAE_WARNING_THRESHOLD:
            return STATUS_WARNING
        return STATUS_HEALTHY

    def evaluate_uncertainty_status(self, coverage: float | None) -> str:
        if coverage is None or coverage < UNCERTAINTY_COVERAGE_WARNING_MIN:
            return STATUS_CRITICAL
        if (
            coverage < UNCERTAINTY_COVERAGE_HEALTHY_MIN
            or coverage > UNCERTAINTY_COVERAGE_HEALTHY_MAX
        ):
            return STATUS_WARNING
        return STATUS_HEALTHY

    def evaluate_risk_summary(
        self,
        forecast_run_id: str,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not rows:
            return _section(
                STATUS_CRITICAL,
                {"latest_forecast_run_id": forecast_run_id},
                errors=["Latest forecast contains no risk rows."],
            )
        frame = pd.DataFrame(rows)
        frame["forecast_ptf"] = pd.to_numeric(frame["forecast_ptf"], errors="coerce")
        frame["interval_width_95"] = pd.to_numeric(
            frame["interval_width_95"],
            errors="coerce",
        )
        counts = frame["risk_level"].value_counts().to_dict()
        high_risk_hours = int(counts.get("HIGH", 0))
        status = STATUS_HEALTHY
        warnings: list[str] = []
        errors: list[str] = []
        if high_risk_hours >= RISK_CRITICAL_HIGH_HOURS:
            status = STATUS_CRITICAL
            errors.append(f"{high_risk_hours} high-risk hours in latest forecast.")
        elif high_risk_hours >= RISK_WARNING_HIGH_HOURS:
            status = STATUS_WARNING
            warnings.append(f"{high_risk_hours} high-risk hours in latest forecast.")
        return _section(
            status,
            {
                "latest_forecast_run_id": forecast_run_id,
                "risk_level_counts": {
                    "LOW": int(counts.get("LOW", 0)),
                    "MEDIUM": int(counts.get("MEDIUM", 0)),
                    "HIGH": high_risk_hours,
                },
                "mean_forecast": _optional_float(frame["forecast_ptf"].mean()),
                "min_forecast": _optional_float(frame["forecast_ptf"].min()),
                "max_forecast": _optional_float(frame["forecast_ptf"].max()),
                "mean_interval_width": _optional_float(
                    frame["interval_width_95"].mean()
                ),
                "high_risk_hours": high_risk_hours,
            },
            warnings=warnings,
            errors=errors,
        )

    def store_snapshot(self, snapshot: dict[str, Any]) -> None:
        statement = text(
            """
            INSERT INTO monitoring_snapshots (
                snapshot_id,
                status,
                created_at,
                data_freshness,
                data_quality,
                pipeline_health,
                forecast_health,
                model_quality,
                uncertainty_quality,
                risk_summary,
                warnings,
                errors
            )
            VALUES (
                :snapshot_id,
                :status,
                :created_at,
                CAST(:data_freshness AS JSONB),
                CAST(:data_quality AS JSONB),
                CAST(:pipeline_health AS JSONB),
                CAST(:forecast_health AS JSONB),
                CAST(:model_quality AS JSONB),
                CAST(:uncertainty_quality AS JSONB),
                CAST(:risk_summary AS JSONB),
                CAST(:warnings AS JSONB),
                CAST(:errors AS JSONB)
            )
            """
        )
        with self.session_factory() as session:
            session.execute(
                statement,
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "status": snapshot["status"],
                    "created_at": snapshot["created_at"],
                    "data_freshness": json.dumps(
                        _json_ready(snapshot["data_freshness"])
                    ),
                    "data_quality": json.dumps(_json_ready(snapshot["data_quality"])),
                    "pipeline_health": json.dumps(
                        _json_ready(snapshot["pipeline_health"])
                    ),
                    "forecast_health": json.dumps(
                        _json_ready(snapshot["forecast_health"])
                    ),
                    "model_quality": json.dumps(_json_ready(snapshot["model_quality"])),
                    "uncertainty_quality": json.dumps(
                        _json_ready(snapshot["uncertainty_quality"])
                    ),
                    "risk_summary": json.dumps(_json_ready(snapshot["risk_summary"])),
                    "warnings": json.dumps(_json_ready(snapshot["warnings"])),
                    "errors": json.dumps(_json_ready(snapshot["errors"])),
                },
            )
            session.commit()

    def get_latest_snapshot(self) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT *
                    FROM monitoring_snapshots
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).mappings().one_or_none()
        return _snapshot_row_to_dict(row) if row is not None else {}

    def list_snapshots(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self.session_factory() as session:
            rows = [
                _snapshot_row_to_dict(row)
                for row in session.execute(
                    text(
                        """
                        SELECT *
                        FROM monitoring_snapshots
                        ORDER BY created_at DESC, id DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                ).mappings()
            ]
        return rows

    def get_compact_status(self) -> dict[str, Any]:
        snapshot = self.get_latest_snapshot()
        if not snapshot:
            return {}
        return {
            "status": snapshot["status"],
            "created_at": snapshot["created_at"],
            "warnings": snapshot.get("warnings", [])[:10],
            "errors": snapshot.get("errors", [])[:10],
            "latest_pipeline_status": snapshot.get("pipeline_health", {}).get(
                "latest_status"
            ),
            "latest_forecast_run_id": snapshot.get("forecast_health", {}).get(
                "latest_forecast_run_id"
            ),
            "latest_data_timestamp": snapshot.get("data_freshness", {}).get(
                "max_timestamp"
            ),
            "latest_model_metrics": {
                "selected_model": snapshot.get("model_quality", {}).get(
                    "latest_selected_model"
                ),
                "mae": snapshot.get("model_quality", {}).get("mae"),
                "rmse": snapshot.get("model_quality", {}).get("rmse"),
                "r2": snapshot.get("model_quality", {}).get("r2"),
            },
        }


def _section(
    status: str,
    payload: dict[str, Any],
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        **payload,
        "status": status,
        "warnings": warnings or [],
        "errors": errors or [],
    }


def _collect_messages(sections: list[dict[str, Any]], key: str) -> list[str]:
    messages: list[str] = []
    for section in sections:
        messages.extend(str(message) for message in section.get(key, []) or [])
    return messages


def _snapshot_row_to_dict(row: Any) -> dict[str, Any]:
    return _json_ready(dict(row)) if row is not None else {}


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _as_aware_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=ISTANBUL_TIMEZONE)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(ISTANBUL_TIMEZONE)
    return timestamp.to_pydatetime()


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
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
