import logging
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import SessionLocal, engine
from data_pipeline.validation.features import (
    FeatureValidationResult,
    validate_ptf_feature_frame,
)

logger = logging.getLogger(__name__)

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")
MAX_HISTORY_HOURS = 30 * 24
LAG_HOURS = (1, 2, 3, 24, 48, 72, 168)
FEATURE_COLUMNS = [
    "timestamp",
    "target_ptf",
    "hour",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_peak_hour",
    "is_business_hour",
    "season",
    "ptf_lag_1",
    "ptf_lag_2",
    "ptf_lag_3",
    "ptf_lag_24",
    "ptf_lag_48",
    "ptf_lag_72",
    "ptf_lag_168",
    "ptf_24h_mean",
    "ptf_24h_std",
    "ptf_24h_min",
    "ptf_24h_max",
    "ptf_7d_mean",
    "ptf_7d_std",
    "ptf_7d_min",
    "ptf_7d_max",
    "ptf_30d_mean",
    "ptf_30d_std",
    "ptf_diff_1",
    "ptf_diff_24",
    "ptf_pct_change_1",
    "ptf_pct_change_24",
    "feature_version",
]


class PtfFeatureService:
    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self.session_factory = session_factory

    def load_ptf_data(
        self,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: dict[str, datetime] = {}
        if start_date is not None:
            clauses.append('"timestamp" >= :start_timestamp')
            parameters["start_timestamp"] = _normalize_boundary(
                start_date,
                is_end=False,
            )
        if end_date is not None:
            clauses.append('"timestamp" <= :end_timestamp')
            parameters["end_timestamp"] = _normalize_boundary(
                end_date,
                is_end=True,
            )

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = text(
            f"""
            SELECT "timestamp", ptf_tl
            FROM ptf_hourly
            {where_clause}
            ORDER BY "timestamp"
            """
        )
        with engine.connect() as connection:
            dataframe = pd.read_sql_query(query, connection, params=parameters)

        if dataframe.empty:
            return pd.DataFrame(
                {
                    "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                    "ptf_tl": pd.Series(dtype="float64"),
                }
            )

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            utc=True,
        )
        dataframe["ptf_tl"] = pd.to_numeric(dataframe["ptf_tl"], errors="coerce")
        return dataframe.sort_values("timestamp").reset_index(drop=True)

    def build_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "ptf_tl"}
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(
                f"Missing required PTF columns: {', '.join(sorted(missing))}"
            )

        source = dataframe.copy()
        source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)
        source["ptf_tl"] = pd.to_numeric(source["ptf_tl"], errors="coerce")
        source = source.sort_values("timestamp").reset_index(drop=True)

        features = pd.DataFrame(index=source.index)
        features["timestamp"] = source["timestamp"]
        features["target_ptf"] = source["ptf_tl"]

        local_timestamp = source["timestamp"].dt.tz_convert(ISTANBUL_TIMEZONE)
        features["hour"] = local_timestamp.dt.hour.astype("int64")
        features["day_of_week"] = local_timestamp.dt.dayofweek.astype("int64")
        features["day_of_month"] = local_timestamp.dt.day.astype("int64")
        features["day_of_year"] = local_timestamp.dt.dayofyear.astype("int64")
        features["week_of_year"] = (
            local_timestamp.dt.isocalendar().week.astype("int64")
        )
        features["month"] = local_timestamp.dt.month.astype("int64")
        features["quarter"] = local_timestamp.dt.quarter.astype("int64")
        features["year"] = local_timestamp.dt.year.astype("int64")
        features["is_weekend"] = features["day_of_week"] >= 5
        features["is_month_start"] = local_timestamp.dt.is_month_start
        features["is_month_end"] = local_timestamp.dt.is_month_end
        features["is_peak_hour"] = (
            (features["day_of_week"] < 5)
            & features["hour"].between(8, 19)
        )
        features["is_business_hour"] = (
            (features["day_of_week"] < 5)
            & features["hour"].between(9, 17)
        )
        features["season"] = features["month"].map(_season_for_month)

        target = features["target_ptf"]
        for lag in LAG_HOURS:
            features[f"ptf_lag_{lag}"] = target.shift(lag)

        past_target = target.shift(1)
        _add_rolling_features(features, past_target, window=24, label="24h")
        _add_rolling_features(features, past_target, window=7 * 24, label="7d")
        features["ptf_30d_mean"] = past_target.rolling(
            30 * 24,
            min_periods=30 * 24,
        ).mean()
        features["ptf_30d_std"] = past_target.rolling(
            30 * 24,
            min_periods=30 * 24,
        ).std()

        features["ptf_diff_1"] = target - features["ptf_lag_1"]
        features["ptf_diff_24"] = target - features["ptf_lag_24"]
        features["ptf_pct_change_1"] = target.pct_change(
            periods=1,
            fill_method=None,
        )
        features["ptf_pct_change_24"] = target.pct_change(
            periods=24,
            fill_method=None,
        )
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        return features

    def validate_features(
        self,
        dataframe: pd.DataFrame,
        expected_rows: int | None = None,
    ) -> FeatureValidationResult:
        return validate_ptf_feature_frame(dataframe, expected_rows=expected_rows)

    def upsert_features(
        self,
        dataframe: pd.DataFrame,
        session: Session | None = None,
        batch_size: int = 1000,
    ) -> int:
        if dataframe.empty:
            return 0

        columns = FEATURE_COLUMNS
        missing = set(columns).difference(dataframe.columns)
        if missing:
            raise ValueError(
                f"Missing columns for feature upsert: {', '.join(sorted(missing))}"
            )

        update_columns = [
            column
            for column in columns
            if column not in {"timestamp", "created_at", "updated_at"}
        ]
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        bound_values = ", ".join(f":{column}" for column in columns)
        update_clause = ", ".join(
            f'"{column}" = EXCLUDED."{column}"' for column in update_columns
        )
        statement = text(
            f"""
            INSERT INTO features_ptf_hourly ({quoted_columns})
            VALUES ({bound_values})
            ON CONFLICT ("timestamp") DO UPDATE SET
                {update_clause},
                updated_at = NOW()
            """
        )
        records = [
            {column: _python_value(row[column]) for column in columns}
            for row in dataframe[columns].to_dict(orient="records")
        ]

        owns_session = session is None
        database_session = session or self.session_factory()
        affected_rows = 0
        try:
            for offset in range(0, len(records), batch_size):
                batch = records[offset : offset + batch_size]
                result = database_session.execute(statement, batch)
                affected_rows += (
                    result.rowcount if result.rowcount >= 0 else len(batch)
                )
            database_session.commit()
            return affected_rows
        except SQLAlchemyError:
            database_session.rollback()
            raise
        finally:
            if owns_session:
                database_session.close()

    def build_and_store_features(
        self,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
        feature_version: str = "v1",
    ) -> dict[str, Any]:
        if not feature_version.strip():
            raise ValueError("feature_version must not be empty")

        selected_start = (
            _normalize_boundary(start_date, is_end=False)
            if start_date is not None
            else None
        )
        selected_end = (
            _normalize_boundary(end_date, is_end=True)
            if end_date is not None
            else None
        )
        if selected_start and selected_end and selected_end < selected_start:
            raise ValueError("end_date must be on or after start_date")

        load_start = (
            selected_start - timedelta(hours=MAX_HISTORY_HOURS)
            if selected_start is not None
            else None
        )
        source_with_history = self.load_ptf_data(load_start, selected_end)
        built_with_history = self.build_features(source_with_history)

        source_mask = pd.Series(True, index=source_with_history.index)
        if selected_start is not None:
            source_mask &= source_with_history["timestamp"] >= pd.Timestamp(
                selected_start
            )
        if selected_end is not None:
            source_mask &= source_with_history["timestamp"] <= pd.Timestamp(
                selected_end
            )
        source_rows = int(source_mask.sum())

        selected_mask = pd.Series(True, index=built_with_history.index)
        if selected_start is not None:
            selected_mask &= built_with_history["timestamp"] >= pd.Timestamp(
                selected_start
            )
        if selected_end is not None:
            selected_mask &= built_with_history["timestamp"] <= pd.Timestamp(
                selected_end
            )
        features = built_with_history.loc[selected_mask].copy().reset_index(drop=True)
        features["feature_version"] = feature_version

        validation = self.validate_features(features, expected_rows=source_rows)
        summary: dict[str, Any] = {
            "source_rows": source_rows,
            "feature_rows_built": len(features),
            "feature_rows_inserted_or_updated": 0,
            "min_timestamp": (
                features["timestamp"].min().isoformat() if not features.empty else None
            ),
            "max_timestamp": (
                features["timestamp"].max().isoformat() if not features.empty else None
            ),
            "feature_version": feature_version,
            "warnings": validation.warnings,
            "errors": validation.errors,
        }
        if features.empty:
            summary["errors"].append("No PTF source rows found for the selected range")
            return summary
        if validation.errors:
            return summary

        summary["feature_rows_inserted_or_updated"] = self.upsert_features(features)
        return summary

    def get_status(self) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS total_rows,
                        MIN("timestamp") AS min_timestamp,
                        MAX("timestamp") AS max_timestamp,
                        MAX(updated_at) AS latest_updated_at,
                        COALESCE(
                            ARRAY_AGG(DISTINCT feature_version)
                                FILTER (WHERE feature_version IS NOT NULL),
                            ARRAY[]::TEXT[]
                        ) AS feature_versions
                    FROM features_ptf_hourly
                    """
                )
            ).mappings().one()
            return dict(row)


def _normalize_boundary(value: date | datetime, is_end: bool) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=ISTANBUL_TIMEZONE)
        return value.astimezone(ISTANBUL_TIMEZONE)
    boundary = time(23, 59, 59) if is_end else time.min
    return datetime.combine(value, boundary, tzinfo=ISTANBUL_TIMEZONE)


def _season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _add_rolling_features(
    features: pd.DataFrame,
    past_target: pd.Series,
    window: int,
    label: str,
) -> None:
    rolling = past_target.rolling(window=window, min_periods=window)
    features[f"ptf_{label}_mean"] = rolling.mean()
    features[f"ptf_{label}_std"] = rolling.std()
    features[f"ptf_{label}_min"] = rolling.min()
    features[f"ptf_{label}_max"] = rolling.max()


def _python_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value
