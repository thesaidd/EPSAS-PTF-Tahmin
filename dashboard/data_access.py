import os
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


def get_db_config() -> DatabaseConfig:
    return DatabaseConfig(
        host=os.getenv("POSTGRES_HOST", "db"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "pepias"),
        password=os.getenv("POSTGRES_PASSWORD", "pepias"),
        database=os.getenv("POSTGRES_DB", "pepias"),
    )


@contextmanager
def get_db_connection():
    config = get_db_config()
    connection = psycopg.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        dbname=config.database,
        row_factory=dict_row,
    )
    try:
        yield connection
    finally:
        connection.close()


def load_decision_runs() -> pd.DataFrame:
    query = """
        SELECT
            decision_run_id,
            gpr_run_id,
            xgboost_training_run_id,
            model_version,
            selected_model,
            evaluation_start,
            evaluation_end,
            mae,
            interval_coverage_95,
            created_at
        FROM forecast_decision_metrics
        ORDER BY created_at DESC, id DESC
    """
    rows = _fetch_all(query)
    return pd.DataFrame(rows)


def load_latest_decision_metrics() -> dict[str, Any] | None:
    runs = load_decision_runs()
    if runs.empty:
        return None
    return load_decision_metrics(str(runs.iloc[0]["decision_run_id"]))


def load_latest_day_ahead_forecast() -> tuple[dict[str, Any] | None, pd.DataFrame]:
    latest_query = """
        SELECT forecast_run_id
        FROM day_ahead_forecasts
        ORDER BY generated_at DESC, id DESC
        LIMIT 1
    """
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(latest_query)
            latest = cursor.fetchone()

    if latest is None:
        return None, pd.DataFrame()

    forecast_run_id = str(latest["forecast_run_id"])
    rows = _fetch_all(
        """
        SELECT
            forecast_run_id,
            target_date,
            "timestamp",
            horizon_hour,
            selected_model,
            xgboost_prediction,
            residual_mean,
            residual_std,
            forecast_ptf,
            lower_bound_95,
            upper_bound_95,
            interval_width_95,
            risk_level,
            xgboost_training_run_id,
            gpr_run_id,
            decision_run_id,
            model_version,
            generation_method,
            warnings,
            generated_at
        FROM day_ahead_forecasts
        WHERE forecast_run_id = %(forecast_run_id)s
        ORDER BY "timestamp"
        """,
        {"forecast_run_id": forecast_run_id},
    )
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return None, dataframe

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
    for column in [
        "xgboost_prediction",
        "residual_mean",
        "residual_std",
        "forecast_ptf",
        "lower_bound_95",
        "upper_bound_95",
        "interval_width_95",
    ]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    risk_counts = dataframe["risk_level"].value_counts().to_dict()
    summary = {
        "forecast_run_id": forecast_run_id,
        "target_date": dataframe.iloc[0]["target_date"],
        "generated_at": dataframe.iloc[0]["generated_at"],
        "model_version": dataframe.iloc[0]["model_version"],
        "selected_model": dataframe.iloc[0]["selected_model"],
        "rows": len(dataframe),
        "mean_forecast": dataframe["forecast_ptf"].mean(),
        "min_forecast": dataframe["forecast_ptf"].min(),
        "max_forecast": dataframe["forecast_ptf"].max(),
        "mean_interval_width": dataframe["interval_width_95"].mean(),
        "risk_level_counts": {
            "LOW": int(risk_counts.get("LOW", 0)),
            "MEDIUM": int(risk_counts.get("MEDIUM", 0)),
            "HIGH": int(risk_counts.get("HIGH", 0)),
        },
    }
    return summary, dataframe


def load_decision_metrics(decision_run_id: str) -> dict[str, Any] | None:
    query = """
        SELECT *
        FROM forecast_decision_metrics
        WHERE decision_run_id = %(decision_run_id)s
        ORDER BY created_at DESC, id DESC
        LIMIT 1
    """
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, {"decision_run_id": decision_run_id})
            row = cursor.fetchone()
    return dict(row) if row is not None else None


def build_decision_predictions_query(
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
    risk_levels: Sequence[str] | None = None,
    limit: int | None = None,
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = ["decision_run_id = %(decision_run_id)s"]
    parameters: dict[str, Any] = {}
    if start_date is not None:
        clauses.append('"timestamp" >= %(start_date)s')
        parameters["start_date"] = _start_boundary(start_date)
    if end_date is not None:
        clauses.append('"timestamp" <= %(end_date)s')
        parameters["end_date"] = _end_boundary(end_date)
    if risk_levels:
        clauses.append("risk_level = ANY(%(risk_levels)s)")
        parameters["risk_levels"] = list(risk_levels)

    limit_clause = ""
    if limit is not None and limit > 0:
        limit_clause = "LIMIT %(limit)s"
        parameters["limit"] = int(limit)

    query = f"""
        SELECT
            "timestamp",
            selected_model,
            xgboost_prediction,
            gpr_corrected_prediction,
            selected_prediction,
            actual,
            residual_mean,
            residual_std,
            lower_bound_95,
            upper_bound_95,
            interval_width_95,
            risk_level,
            error,
            absolute_error,
            percentage_error
        FROM forecast_decision_predictions
        WHERE {' AND '.join(clauses)}
        ORDER BY "timestamp"
        {limit_clause}
    """
    return query, parameters


def load_decision_predictions(
    decision_run_id: str,
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
    risk_levels: Sequence[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    query, parameters = build_decision_predictions_query(
        start_date=start_date,
        end_date=end_date,
        risk_levels=risk_levels,
        limit=limit,
    )
    parameters["decision_run_id"] = decision_run_id
    rows = _fetch_all(query, parameters)
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
    return dataframe


def _start_boundary(value: date | datetime) -> date | datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, time.min)


def _end_boundary(value: date | datetime) -> date | datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, time.max)


def _fetch_all(query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters or {})
            return [dict(row) for row in cursor.fetchall()]
